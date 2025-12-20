import os
import pandas as pd
import numpy as np
import torch
import src.GeometryUtils as gu
from src.FileProcessor import FileProcessor
import gc
import warnings
import tempfile
import shutil
from pathlib import Path
import subprocess
from abc import ABC, abstractmethod

torch.set_default_dtype(torch.float64)
np.seterr(under='ignore')

# ============================================================================
# 1. DATA SOURCE LAYER
# ============================================================================

class FileSystemScanner:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
    
    def discover_files(self, intersection_type):
        intersection_folder = os.path.join(self.raw_data_path, intersection_type)
        
        if not os.path.exists(intersection_folder):
            print(f"  Warning: Folder not found: {intersection_folder}")
            return []
        
        return [os.path.join(intersection_folder, f) 
                for f in os.listdir(intersection_folder) 
                if f.endswith(".csv")]

class DataInventory:
    def __init__(self, file_processor):
        self.file_processor = file_processor
    
    def count_raw_data(self, raw_data_path):
        total = 0
        print("Counting raw data files:")
        
        for intersection_type in os.listdir(raw_data_path):
            folder_path = os.path.join(raw_data_path, intersection_type)
            if os.path.isdir(folder_path):
                folder_total = 0
                for csv_file in os.listdir(folder_path):
                    if csv_file.endswith('.csv'):
                        file_path = os.path.join(folder_path, csv_file)
                        try:
                            count = self.file_processor.count_samples_in_file(file_path)
                            folder_total += count
                        except Exception as e:
                            print(f"  Error reading {csv_file}: {e}")
                total += folder_total
                print(f"  {intersection_type} total: {folder_total:,} rows")
        
        print(f"Total raw data: {total:,} rows")
        return total

# ============================================================================
# 2. QUALITY CONTROL LAYER
# ============================================================================

class DataTypeNormalizer:
    def __init__(self, config):
        self.config = config
    
    def basic_conversion_only(self, df):
        """Minimal type conversion without scaling - for combining raw files"""
        if df.empty:
            return df
        
        df = df.copy()
        
        try:
            coordinate_columns = df.columns[:-2]
            for col in coordinate_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
            if 'IntersectionVolume' in df.columns:
                df['IntersectionVolume'] = pd.to_numeric(df['IntersectionVolume'], errors='coerce').astype('float64')
            
            if 'HasIntersection' in df.columns:
                df['HasIntersection'] = pd.to_numeric(df['HasIntersection'], errors='coerce').astype('int32')
            
            all_nan_mask = df.isna().all(axis=1)
            if all_nan_mask.any():
                initial_rows = len(df)
                df = df[~all_nan_mask]
                removed_rows = initial_rows - len(df)
                if removed_rows > 0:
                    print(f"  Removed {removed_rows} completely empty rows")
            
            df = df.fillna(0)
            
        except Exception as e:
            print(f"  Warning: Type conversion error: {e}")
            return pd.DataFrame()
        return df

    def normalize_data_types_with_scaling(self, df):
        """Apply type conversion AND volume scaling - for final output"""
        if df.empty:
            return df
        
        df = df.copy()
        
        try:
            coordinate_columns = df.columns[:-2]
            for col in coordinate_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
            if 'IntersectionVolume' in df.columns:
                df['IntersectionVolume'] = pd.to_numeric(df['IntersectionVolume'], errors='coerce').astype('float64')
                
                # Apply scaling here
                volume_scale_factor = self.config.get("volume_scale_factor", 1.0)
                if volume_scale_factor != 1.0:
                    df['IntersectionVolume'] = df['IntersectionVolume'] * float(volume_scale_factor)
            
            if 'HasIntersection' in df.columns:
                df['HasIntersection'] = pd.to_numeric(df['HasIntersection'], errors='coerce').astype('int32')
            
            all_nan_mask = df.isna().all(axis=1)
            if all_nan_mask.any():
                initial_rows = len(df)
                df = df[~all_nan_mask]
                removed_rows = initial_rows - len(df)
                if removed_rows > 0:
                    print(f"  Removed {removed_rows} completely empty rows")
            
            df = df.fillna(0)
            
        except Exception as e:
            print(f"  Warning: Type conversion error: {e}")
            return pd.DataFrame()
        return df

# ============================================================================
# 3. DISTRIBUTION STRATEGY LAYER
# ============================================================================

class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, source_file, intersection_type, train_target, val_target, temp_dir):
        pass

class RegularSampler(SamplingStrategy):
    def __init__(self, file_processor, normalizer):
        self.file_processor = file_processor
        self.normalizer = normalizer
    
    def sample(self, source_file, intersection_type, train_target, val_target, temp_dir):
        total_available = self.file_processor.count_samples_in_file(source_file)
        
        if total_available == 0:
            return None, None, 0, 0
        
        total_needed = train_target + val_target
        if total_needed > total_available:
            scale_factor = total_available / total_needed
            train_target = int(train_target * scale_factor)
            val_target = int(val_target * scale_factor)
            print(f"  Scaled down to {train_target} train + {val_target} val samples")
        
        train_file = os.path.join(temp_dir, f"{intersection_type}_train.csv")
        val_file = os.path.join(temp_dir, f"{intersection_type}_val.csv")
        
        train_collected, val_collected = self._stream_sample_file(
            source_file, train_file, val_file, train_target, val_target
        )
        
        return (train_file if train_collected > 0 else None,
                val_file if val_collected > 0 else None,
                train_collected, val_collected)
    
    def _stream_sample_file(self, source_file, train_file, val_file, train_target, val_target):
        chunk_size = 10_000
        train_collected = 0
        val_collected = 0
        first_train_chunk = True
        first_val_chunk = True
        
        for chunk_df in pd.read_csv(source_file, chunksize=chunk_size):
            if train_collected >= train_target and val_collected >= val_target:
                break
            
            # Apply scaling here during sampling
            chunk_df = self.normalizer.normalize_data_types_with_scaling(chunk_df)
            chunk_df = chunk_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            if train_collected < train_target:
                train_needed = min(train_target - train_collected, len(chunk_df))
                train_chunk = chunk_df.iloc[:train_needed].copy()
                train_chunk = self._clean_sample_data(train_chunk)
                
                train_chunk.to_csv(train_file, mode='a', header=first_train_chunk, index=False,
                                float_format='%.17g')
                first_train_chunk = False
                train_collected += len(train_chunk)
                
                remaining_chunk = chunk_df.iloc[train_needed:].copy()
            else:
                remaining_chunk = chunk_df.copy()
            
            if val_collected < val_target and len(remaining_chunk) > 0:
                val_needed = min(val_target - val_collected, len(remaining_chunk))
                val_chunk = remaining_chunk.iloc[:val_needed].copy()
                val_chunk = self._clean_sample_data(val_chunk)
                
                val_chunk.to_csv(val_file, mode='a', header=first_val_chunk, index=False,
                                float_format='%.17g')
                first_val_chunk = False
                val_collected += len(val_chunk)
            
            del chunk_df
            gc.collect()
        
        return train_collected, val_collected
    
    def _clean_sample_data(self, df):
        if df.empty:
            return df
        
        df = df.copy()
        df = df.dropna(how='all')
        df = df.fillna(0)
        return df

class UniformVolumeSampler(SamplingStrategy):
    def __init__(self, config, normalizer, file_processor):
        self.config = config
        self.normalizer = normalizer
        self.file_processor = file_processor

    def sample(self, source_file, intersection_type, train_target, val_target, temp_dir):
        print(f"      Applying GUARANTEED uniform volume distribution sampling")
        
        # Get volume binning config
        volume_binning_config = self.config.get("volume_binning", {})
        volume_range = volume_binning_config.get("volume_range")
        n_bins = volume_binning_config.get("n_bins", 10)
        
        if not volume_range:
            print(f"        No volume_range specified, falling back to regular sampling")
            regular_sampler = RegularSampler(self.file_processor, self.normalizer)
            return regular_sampler.sample(source_file, intersection_type, train_target, val_target, temp_dir)
        
        config_min_raw = float(volume_range[0])
        config_max_raw = float(volume_range[1])
        
        print(f"        Volume range: [{config_min_raw:.8e}, {config_max_raw:.8e}]")
        print(f"        Bins: {n_bins} (uniform volume distribution)")
        print(f"        Targets: {train_target} train + {val_target} val (GUARANTEED)")
        
        # Create uniform volume bin edges
        bin_edges = np.linspace(config_min_raw, config_max_raw, n_bins + 1)
        
        print(f"        Uniform volume bins:")
        for i in range(n_bins):
            volume_width = bin_edges[i+1] - bin_edges[i]
            print(f"          Bin {i+1:2d}: [{bin_edges[i]:.8e}, {bin_edges[i+1]:.8e}] (width: {volume_width:.8e})")
        
        train_file = os.path.join(temp_dir, f"{intersection_type}_train.csv")
        val_file = os.path.join(temp_dir, f"{intersection_type}_val.csv")
        
        # Guaranteed uniform sampling
        train_count, val_count = self._guaranteed_uniform_volume_sampling(
            source_file, train_file, val_file, train_target, val_target, 
            n_bins, bin_edges, (config_min_raw, config_max_raw)
        )
        
        if train_count != train_target or val_count != val_target:
            print(f"        WARNING: Did not achieve exact targets: got {train_count}+{val_count}, wanted {train_target}+{val_target}")
        
        return train_file, val_file, train_count, val_count

    def _guaranteed_uniform_volume_sampling(self, source_file, train_file, val_file, 
                                          train_target, val_target, n_bins, bin_edges, volume_range):
        """Guaranteed uniform volume sampling with exact sample counts"""
        
        total_target = train_target + val_target
        samples_per_bin = total_target // n_bins
        remainder = total_target % n_bins
        train_ratio = train_target / total_target if total_target > 0 else 0.9
        
        # Calculate exact targets per bin
        bin_targets = []
        for i in range(n_bins):
            total_in_bin = samples_per_bin + (1 if i < remainder else 0)
            train_in_bin = int(total_in_bin * train_ratio)
            val_in_bin = total_in_bin - train_in_bin
            bin_targets.append((train_in_bin, val_in_bin))
        
        print(f"        Exact targets per bin:")
        for i, (train_count, val_count) in enumerate(bin_targets):
            print(f"          Bin {i+1:2d}: {train_count} train + {val_count} val = {train_count + val_count} total")
        
        # Phase 1: Collect samples from each bin
        train_bin_samples = [[] for _ in range(n_bins)]
        val_bin_samples = [[] for _ in range(n_bins)]
        
        chunk_size = 10_000
        config_min, config_max = volume_range
        
        print(f"        Phase 1: Collecting samples from volume range...")
        
        try:
            for chunk_df in pd.read_csv(source_file, chunksize=chunk_size):
                if 'HasIntersection' not in chunk_df.columns or 'IntersectionVolume' not in chunk_df.columns:
                    continue
                
                # Filter for intersecting samples in volume range
                intersecting_mask = chunk_df['HasIntersection'] == 1
                volume_mask = (chunk_df['IntersectionVolume'] >= config_min) & (chunk_df['IntersectionVolume'] <= config_max)
                valid_mask = intersecting_mask & volume_mask
                
                valid_df = chunk_df[valid_mask].copy()
                if len(valid_df) == 0:
                    continue
                
                # Shuffle samples
                valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Assign to bins
                for _, row in valid_df.iterrows():
                    volume = row['IntersectionVolume']
                    
                    # Find bin (ensure last bin includes max value)
                    bin_idx = np.searchsorted(bin_edges[1:], volume, side='right')
                    bin_idx = min(bin_idx, n_bins - 1)
                    
                    train_target_bin, val_target_bin = bin_targets[bin_idx]
                    sample_data = row.to_dict()
                    
                    # Add to train if needed
                    if len(train_bin_samples[bin_idx]) < train_target_bin:
                        train_bin_samples[bin_idx].append(sample_data)
                    # Add to val if needed
                    elif len(val_bin_samples[bin_idx]) < val_target_bin:
                        val_bin_samples[bin_idx].append(sample_data)
                
                # Check if all bins are full
                all_full = all(
                    len(train_bin_samples[i]) >= bin_targets[i][0] and 
                    len(val_bin_samples[i]) >= bin_targets[i][1]
                    for i in range(n_bins)
                )
                
                if all_full:
                    print(f"        All bins filled successfully")
                    break
                
                del chunk_df, valid_df
                gc.collect()
        
        except Exception as e:
            print(f"        Error during sampling: {e}")
            return 0, 0
        
        # Phase 2: Handle unfilled bins with strategic oversampling
        print(f"        Phase 2: Ensuring all bins are filled...")
        
        for bin_idx in range(n_bins):
            train_target_bin, val_target_bin = bin_targets[bin_idx]
            train_current = len(train_bin_samples[bin_idx])
            val_current = len(val_bin_samples[bin_idx])
            
            train_needed = train_target_bin - train_current
            val_needed = val_target_bin - val_current
            
            if train_needed > 0 or val_needed > 0:
                print(f"          Bin {bin_idx+1} needs {train_needed} train + {val_needed} val samples")
                
                # Collect samples from neighboring bins for oversampling
                source_samples = []
                
                # Try adjacent bins first
                for offset in [1, -1, 2, -2]:
                    neighbor_idx = bin_idx + offset
                    if 0 <= neighbor_idx < n_bins:
                        source_samples.extend(train_bin_samples[neighbor_idx])
                        source_samples.extend(val_bin_samples[neighbor_idx])
                
                # If still not enough, use all bins
                if len(source_samples) < (train_needed + val_needed):
                    source_samples = []
                    for i in range(n_bins):
                        source_samples.extend(train_bin_samples[i])
                        source_samples.extend(val_bin_samples[i])
                
                if len(source_samples) > 0:
                    # Oversample to fill the bin
                    needed_total = train_needed + val_needed
                    
                    if len(source_samples) >= needed_total:
                        selected_indices = np.random.choice(len(source_samples), needed_total, replace=False)
                    else:
                        selected_indices = np.random.choice(len(source_samples), needed_total, replace=True)
                    
                    selected_samples = [source_samples[i] for i in selected_indices]
                    
                    # Add to train/val
                    for i, sample in enumerate(selected_samples):
                        if i < train_needed:
                            train_bin_samples[bin_idx].append(sample)
                        else:
                            val_bin_samples[bin_idx].append(sample)
                    
                    print(f"            Added {len(selected_samples)} oversampled samples")
        
        # Phase 3: Write samples with scaling
        print(f"        Phase 3: Writing samples with scaling...")
        
        train_collected = self._write_samples_with_scaling(train_bin_samples, train_file)
        val_collected = self._write_samples_with_scaling(val_bin_samples, val_file)
        
        print(f"        Final result: {train_collected} train + {val_collected} val")
        
        return train_collected, val_collected

    def _write_samples_with_scaling(self, bin_samples, output_file):
        """Write samples to file WITH scaling applied"""
        all_samples = []
        for bin_sample_list in bin_samples:
            all_samples.extend(bin_sample_list)
        
        if len(all_samples) == 0:
            pd.DataFrame().to_csv(output_file, index=False)
            return 0
        
        # Convert to DataFrame and apply scaling
        samples_df = pd.DataFrame(all_samples)
        samples_df = self.normalizer.normalize_data_types_with_scaling(samples_df)
        samples_df.to_csv(output_file, index=False, float_format='%.17g')
        
        print(f"        Saved {len(all_samples)} samples with scaling to {os.path.basename(output_file)}")
        return len(all_samples)

class LogUniformVolumeSampler(SamplingStrategy):
    def __init__(self, config, normalizer, file_processor):
        self.config = config
        self.normalizer = normalizer
        self.file_processor = file_processor

    def sample(self, source_file, intersection_type, train_target, val_target, temp_dir):
        print(f"      Applying LOG-UNIFORM volume distribution sampling")
        
        # Get volume binning config
        volume_binning_config = self.config.get("volume_binning", {})
        volume_range = volume_binning_config.get("volume_range")
        n_bins = volume_binning_config.get("n_bins", 10)
        
        if not volume_range:
            print(f"        No volume_range specified, falling back to regular sampling")
            regular_sampler = RegularSampler(self.file_processor, self.normalizer)
            return regular_sampler.sample(source_file, intersection_type, train_target, val_target, temp_dir)
        
        config_min_raw = float(volume_range[0])
        config_max_raw = float(volume_range[1])
        
        # Calculate actual decades for clarity
        n_decades = np.log10(config_max_raw / config_min_raw)
        
        print(f"        Volume range: [{config_min_raw:.8e}, {config_max_raw:.8e}]")
        print(f"        Spans {n_decades:.1f} decades across {n_bins} log-uniform bins")
        print(f"        Targets: {train_target} train + {val_target} val")
        
        # Create LOG-UNIFORM bin edges (equal width in log-space)
        log_min = np.log10(config_min_raw)
        log_max = np.log10(config_max_raw)
        bin_edges = np.logspace(log_min, log_max, n_bins + 1)
        
        print(f"        Log-uniform bins (equal log-space width):")
        log_width = (log_max - log_min) / n_bins
        for i in range(n_bins):
            factor = bin_edges[i+1] / bin_edges[i]
            print(f"          Bin {i+1:2d}: [{bin_edges[i]:.8e}, {bin_edges[i+1]:.8e}] (×{factor:.2f}, log-width: {log_width:.3f})")
        
        train_file = os.path.join(temp_dir, f"{intersection_type}_train.csv")
        val_file = os.path.join(temp_dir, f"{intersection_type}_val.csv")
        
        # Apply log-uniform sampling with geometric augmentation
        train_count, val_count = self._guaranteed_log_uniform_sampling(
            source_file, train_file, val_file, train_target, val_target, 
            n_bins, bin_edges, (config_min_raw, config_max_raw)
        )
        
        return train_file, val_file, train_count, val_count

    def _guaranteed_log_uniform_sampling(self, source_file, train_file, val_file, 
                                       train_target, val_target, n_bins, bin_edges, volume_range):
        """Log-uniform sampling prioritizing real data with geometric augmentation fallbacks"""
        
        total_target = train_target + val_target
        samples_per_bin = total_target // n_bins
        remainder = total_target % n_bins
        train_ratio = train_target / total_target if total_target > 0 else 0.9
        
        # Calculate targets per bin
        bin_targets = []
        for i in range(n_bins):
            total_in_bin = samples_per_bin + (1 if i < remainder else 0)
            train_in_bin = int(total_in_bin * train_ratio)
            val_in_bin = total_in_bin - train_in_bin
            bin_targets.append((train_in_bin, val_in_bin))
        
        print(f"        Target samples per log-bin:")
        for i, (train_count, val_count) in enumerate(bin_targets):
            bin_range = f"[{bin_edges[i]:.1e}, {bin_edges[i+1]:.1e}]"
            print(f"          Bin {i+1:2d} {bin_range}: {train_count} train + {val_count} val")
        
        # Phase 1: Exhaustive collection of real samples per bin
        train_bin_samples = [[] for _ in range(n_bins)]
        val_bin_samples = [[] for _ in range(n_bins)]
        
        chunk_size = 10_000
        config_min, config_max = volume_range
        
        print(f"        Phase 1: Exhaustive collection of real samples...")
        
        try:
            for chunk_df in pd.read_csv(source_file, chunksize=chunk_size):
                if 'HasIntersection' not in chunk_df.columns or 'IntersectionVolume' not in chunk_df.columns:
                    continue
                
                # Filter for valid intersection volumes
                intersecting_mask = chunk_df['HasIntersection'] == 1
                volume_mask = (chunk_df['IntersectionVolume'] >= config_min) & (chunk_df['IntersectionVolume'] <= config_max)
                valid_mask = intersecting_mask & volume_mask
                
                valid_df = chunk_df[valid_mask].copy()
                if len(valid_df) == 0:
                    continue
                
                valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Assign to log bins
                for _, row in valid_df.iterrows():
                    volume = row['IntersectionVolume']
                    
                    # Find bin index
                    bin_idx = np.searchsorted(bin_edges[1:], volume, side='right')
                    bin_idx = min(bin_idx, n_bins - 1)
                    
                    train_target_bin, val_target_bin = bin_targets[bin_idx]
                    sample_data = row.to_dict()
                    
                    # Add to train or val if space available
                    if len(train_bin_samples[bin_idx]) < train_target_bin:
                        train_bin_samples[bin_idx].append(sample_data)
                    elif len(val_bin_samples[bin_idx]) < val_target_bin:
                        val_bin_samples[bin_idx].append(sample_data)
                
                # Early termination if all bins full
                all_full = all(
                    len(train_bin_samples[i]) >= bin_targets[i][0] and 
                    len(val_bin_samples[i]) >= bin_targets[i][1]
                    for i in range(n_bins)
                )
                
                if all_full:
                    print(f"        All log-bins filled with real data!")
                    break
                
                del chunk_df, valid_df
                gc.collect()
        
        except Exception as e:
            print(f"        Error during log-uniform sampling: {e}")
            return 0, 0
        
        # Assess what we've collected vs what we need
        total_real_collected = 0
        bins_needing_augmentation = 0
        empty_bins = []
        
        for bin_idx in range(n_bins):
            train_current = len(train_bin_samples[bin_idx])
            val_current = len(val_bin_samples[bin_idx])
            train_target_bin, val_target_bin = bin_targets[bin_idx]
            
            bin_total = train_current + val_current
            total_real_collected += bin_total
            
            if train_current < train_target_bin or val_current < val_target_bin:
                bins_needing_augmentation += 1
                
                if bin_total == 0:
                    empty_bins.append(bin_idx)
        
        print(f"        Real data collection results:")
        print(f"          Total real samples collected: {total_real_collected}")
        print(f"          Bins needing augmentation: {bins_needing_augmentation}/{n_bins}")
        print(f"          Completely empty bins: {len(empty_bins)}")
        
        # Phase 2: Intelligent augmentation for log-uniform distribution
        if bins_needing_augmentation > 0:
            print(f"        Phase 2: Intelligent geometric augmentation...")
            
            for bin_idx in range(n_bins):
                train_target_bin, val_target_bin = bin_targets[bin_idx]
                train_current = len(train_bin_samples[bin_idx])
                val_current = len(val_bin_samples[bin_idx])
                
                train_needed = train_target_bin - train_current
                val_needed = val_target_bin - val_current
                
                if train_needed > 0 or val_needed > 0:
                    bin_range = f"[{bin_edges[bin_idx]:.2e}, {bin_edges[bin_idx+1]:.2e}]"
                    print(f"          Bin {bin_idx+1} {bin_range} needs {train_needed} train + {val_needed} val")
                    
                    augmented_samples = []
                    
                    # Strategy 1: Use existing samples in this bin for geometric augmentation
                    current_bin_samples = train_bin_samples[bin_idx] + val_bin_samples[bin_idx]
                    if len(current_bin_samples) > 0:
                        needed_from_existing = min(train_needed + val_needed, len(current_bin_samples) * 2)
                        existing_augmented = self._generate_geometric_variants(
                            current_bin_samples, needed_from_existing
                        )
                        augmented_samples.extend(existing_augmented)
                        print(f"            Generated {len(existing_augmented)} variants from existing bin samples")
                    
                    # Strategy 2: Borrow from adjacent bins (preserves volume similarity)
                    if len(augmented_samples) < (train_needed + val_needed):
                        additional_needed = (train_needed + val_needed) - len(augmented_samples)
                        borrowed_samples = self._borrow_from_adjacent_bins(
                            bin_idx, train_bin_samples, val_bin_samples, additional_needed, bin_edges
                        )
                        augmented_samples.extend(borrowed_samples)
                        print(f"            Borrowed {len(borrowed_samples)} samples from adjacent bins")
                    
                    # Strategy 3: Handle completely empty bins by oversampling from nearest populated bin
                    if bin_idx in empty_bins and len(augmented_samples) < (train_needed + val_needed):
                        additional_needed = (train_needed + val_needed) - len(augmented_samples)
                        nearest_bin_samples = self._oversample_from_nearest_populated_bin(
                            bin_idx, train_bin_samples, val_bin_samples, additional_needed, bin_edges
                        )
                        augmented_samples.extend(nearest_bin_samples)
                        print(f"            Oversampled {len(nearest_bin_samples)} from nearest populated bin")
                    
                    # Strategy 4: Cross-bin sampling if still insufficient
                    if len(augmented_samples) < (train_needed + val_needed):
                        additional_needed = (train_needed + val_needed) - len(augmented_samples)
                        cross_bin_samples = self._cross_bin_sampling_with_augmentation(
                            bin_idx, train_bin_samples, val_bin_samples, additional_needed
                        )
                        augmented_samples.extend(cross_bin_samples)
                        print(f"            Cross-bin sampled {len(cross_bin_samples)} with augmentation")
                    
                    # Add augmented samples to fill gaps
                    for i, sample in enumerate(augmented_samples[:train_needed + val_needed]):
                        if i < train_needed:
                            train_bin_samples[bin_idx].append(sample)
                        else:
                            val_bin_samples[bin_idx].append(sample)
                    
                    actual_added = min(len(augmented_samples), train_needed + val_needed)
                    print(f"            Added {actual_added} samples (prioritizing geometric augmentation)")
        else:
            print(f"        Phase 2: Skipped - all bins filled with real data!")
        
        # Phase 3: Write samples
        print(f"        Phase 3: Writing log-uniform samples...")
        
        train_collected = self._write_samples_with_scaling(train_bin_samples, train_file)
        val_collected = self._write_samples_with_scaling(val_bin_samples, val_file)
        
        # Final statistics
        total_augmented = train_collected + val_collected - total_real_collected
        real_percentage = (total_real_collected / (train_collected + val_collected)) * 100 if (train_collected + val_collected) > 0 else 0
        
        print(f"        Final result: {train_collected} train + {val_collected} val (log-uniform)")
        print(f"          Real data: {total_real_collected} ({real_percentage:.1f}%)")
        print(f"          Augmented: {total_augmented} ({100-real_percentage:.1f}%)")
        
        return train_collected, val_collected

    def _generate_geometric_variants(self, samples, needed_count):
        """Generate geometric variants using GeometryUtils functions"""
        if len(samples) == 0 or needed_count <= 0:
            return []
        
        variants = []
        
        # Calculate how many variants we need
        if needed_count <= len(samples):
            selected_indices = np.random.choice(len(samples), needed_count, replace=False)
        else:
            selected_indices = np.random.choice(len(samples), needed_count, replace=True)
        
        for idx in selected_indices:
            original_sample = samples[idx].copy()
            
            # Randomly choose between different geometric augmentations
            augmentation_type = np.random.choice(['swap', 'permute'], p=[0.6, 0.4])
            
            try:
                if augmentation_type == 'swap':
                    # Use GeometryUtils.swap_tetrahedrons
                    augmented_sample = self._apply_geometric_swap(original_sample)
                else:
                    # Use GeometryUtils.permute_points_within_tetrahedrons
                    augmented_sample = self._apply_geometric_permutation(original_sample)
                
                variants.append(augmented_sample)
                
            except Exception as e:
                print(f"            Warning: Geometric augmentation failed: {e}, keeping original")
                variants.append(original_sample.copy())
        
        return variants

    def _apply_geometric_swap(self, sample):
        """Apply tetrahedron swapping using GeometryUtils.swap_tetrahedrons"""
        swapped_sample = sample.copy()
        
        # Extract coordinate columns in proper order
        coordinate_columns = [col for col in sample.keys() if any(col.startswith(prefix) for prefix in ['T1_', 'T2_'])]
        
        if len(coordinate_columns) == 24:
            try:
                # Convert to tensor format expected by GeometryUtils
                coords_array = np.array([sample[col] for col in coordinate_columns])
                coords_tensor = torch.tensor(coords_array, dtype=torch.float64).unsqueeze(0)  # Add batch dimension
                
                # Apply swapping using GeometryUtils
                swapped_tensor = gu.swap_tetrahedrons(coords_tensor)
                swapped_coords = swapped_tensor.squeeze(0).numpy()  # Remove batch dimension
                
                # Update sample with swapped coordinates
                for i, col in enumerate(coordinate_columns):
                    swapped_sample[col] = swapped_coords[i]
                    
            except Exception as e:
                print(f"            Warning: GeometryUtils swap failed: {e}, using manual swap")
                return self._manual_tetrahedron_swap(sample)
        
        return swapped_sample

    def _apply_geometric_permutation(self, sample):
        """Apply vertex permutation using GeometryUtils.permute_points_within_tetrahedrons"""
        permuted_sample = sample.copy()
        
        # Extract coordinate columns
        coordinate_columns = [col for col in sample.keys() if any(col.startswith(prefix) for prefix in ['T1_', 'T2_'])]
        
        if len(coordinate_columns) == 24:
            try:
                # Convert to tensor format expected by GeometryUtils
                coords_array = np.array([sample[col] for col in coordinate_columns])
                coords_tensor = torch.tensor(coords_array, dtype=torch.float64).unsqueeze(0)  # Add batch dimension
                
                # Apply permutation using GeometryUtils
                permuted_tensor = gu.permute_points_within_tetrahedrons(coords_tensor)
                permuted_coords = permuted_tensor.squeeze(0).numpy()  # Remove batch dimension
                
                # Update sample with permuted coordinates
                for i, col in enumerate(coordinate_columns):
                    permuted_sample[col] = permuted_coords[i]
                    
            except Exception as e:
                print(f"            Warning: GeometryUtils permutation failed: {e}, keeping original")
                # Keep original if permutation fails
                pass
        
        return permuted_sample

    def _manual_tetrahedron_swap(self, sample):
        """Manual tetrahedron swap as fallback"""
        swapped_sample = sample.copy()
        
        # Get coordinate column names
        t1_columns = [col for col in sample.keys() if col.startswith('T1_')]
        t2_columns = [col for col in sample.keys() if col.startswith('T2_')]
        
        if len(t1_columns) == 12 and len(t2_columns) == 12:
            # Swap coordinate values: T1 ↔ T2
            for t1_col, t2_col in zip(t1_columns, t2_columns):
                swapped_sample[t1_col] = sample[t2_col]
                swapped_sample[t2_col] = sample[t1_col]
        
        return swapped_sample

    def _borrow_from_adjacent_bins(self, target_bin_idx, train_samples, val_samples, needed_count, bin_edges):
        """Borrow samples from adjacent bins with volume similarity priority"""
        borrowed = []
        
        # Calculate volume similarity weights for adjacent bins
        target_center = np.sqrt(bin_edges[target_bin_idx] * bin_edges[target_bin_idx + 1])  # Geometric mean
        
        adjacent_priorities = []
        for offset in range(1, len(train_samples)):
            for direction in [-1, 1]:
                neighbor_idx = target_bin_idx + (direction * offset)
                if 0 <= neighbor_idx < len(train_samples):
                    neighbor_center = np.sqrt(bin_edges[neighbor_idx] * bin_edges[neighbor_idx + 1])
                    similarity = 1.0 / (1.0 + abs(np.log10(neighbor_center / target_center)))
                    adjacent_priorities.append((neighbor_idx, similarity))
        
        # Sort by similarity (highest first)
        adjacent_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Borrow from most similar bins first
        for neighbor_idx, similarity in adjacent_priorities:
            if len(borrowed) >= needed_count:
                break
                
            available = train_samples[neighbor_idx] + val_samples[neighbor_idx]
            if len(available) == 0:
                continue
                
            # Borrow more from more similar bins
            max_borrow = max(1, int(len(available) * similarity * 0.3))  # Scale by similarity
            to_borrow = min(max_borrow, needed_count - len(borrowed), len(available))
            
            if to_borrow > 0:
                borrowed_indices = np.random.choice(len(available), to_borrow, replace=False)
                for idx in borrowed_indices:
                    borrowed.append(available[idx].copy())
        
        return borrowed

    def _oversample_from_nearest_populated_bin(self, target_bin_idx, train_samples, val_samples, needed_count, bin_edges):
        """Handle completely empty bins by oversampling from nearest populated bin"""
        # Find nearest populated bin
        min_distance = float('inf')
        nearest_bin_idx = None
        
        for bin_idx in range(len(train_samples)):
            if bin_idx == target_bin_idx:
                continue
                
            bin_samples = train_samples[bin_idx] + val_samples[bin_idx]
            if len(bin_samples) > 0:
                distance = abs(bin_idx - target_bin_idx)
                if distance < min_distance:
                    min_distance = distance
                    nearest_bin_idx = bin_idx
        
        if nearest_bin_idx is None:
            print(f"            Warning: No populated bins found for oversampling")
            return []
        
        # Oversample from nearest populated bin
        source_samples = train_samples[nearest_bin_idx] + val_samples[nearest_bin_idx]
        
        if needed_count <= len(source_samples):
            selected_indices = np.random.choice(len(source_samples), needed_count, replace=False)
        else:
            selected_indices = np.random.choice(len(source_samples), needed_count, replace=True)
        
        oversampled = []
        for idx in selected_indices:
            # Apply geometric augmentation to differentiate from original
            original_sample = source_samples[idx].copy()
            augmented_sample = self._apply_geometric_permutation(original_sample)
            oversampled.append(augmented_sample)
        
        return oversampled

    def _cross_bin_sampling_with_augmentation(self, target_bin_idx, train_samples, val_samples, needed_count):
        """Cross-bin sampling with geometric augmentation to increase diversity"""
        # Collect from all bins (excluding target bin)
        all_available = []
        for bin_idx in range(len(train_samples)):
            if bin_idx != target_bin_idx:
                bin_samples = train_samples[bin_idx] + val_samples[bin_idx]
                all_available.extend(bin_samples)
        
        if len(all_available) == 0:
            return []
        
        # Sample and augment
        if needed_count <= len(all_available):
            selected_indices = np.random.choice(len(all_available), needed_count, replace=False)
        else:
            selected_indices = np.random.choice(len(all_available), needed_count, replace=True)
        
        augmented_samples = []
        for idx in selected_indices:
            original_sample = all_available[idx].copy()
            # Apply random geometric augmentation for diversity
            augmentation_type = np.random.choice(['swap', 'permute'])
            
            if augmentation_type == 'swap':
                augmented_sample = self._apply_geometric_swap(original_sample)
            else:
                augmented_sample = self._apply_geometric_permutation(original_sample)
            
            augmented_samples.append(augmented_sample)
        
        return augmented_samples

    def _write_samples_with_scaling(self, bin_samples, output_file):
        """Write samples to file WITH scaling applied"""
        all_samples = []
        for bin_sample_list in bin_samples:
            all_samples.extend(bin_sample_list)
        
        if len(all_samples) == 0:
            pd.DataFrame().to_csv(output_file, index=False)
            return 0
        
        # Convert to DataFrame and apply scaling
        samples_df = pd.DataFrame(all_samples)
        samples_df = self.normalizer.normalize_data_types_with_scaling(samples_df)
        samples_df.to_csv(output_file, index=False, float_format='%.17g')
        
        print(f"        Saved {len(all_samples)} log-uniform samples to {os.path.basename(output_file)}")
        return len(all_samples)

class HybridVolumeSampler(SamplingStrategy):
    def __init__(self, config, normalizer, file_processor):
        # Don't call super().__init__() since SamplingStrategy is an ABC
        self.config = config
        self.normalizer = normalizer
        self.file_processor = file_processor
        self.volume_range = config.get("volume_range", (1e-8, 0.02))
        self.nbins = config.get("number_of_bins", 100)
        self.linear_fraction = config.get("linear_fraction", 0.3)  # 30% linear, 70% log
        self.transition_percentile = config.get("transition_percentile", 70)  # Where to switch from log to linear
        
    def sample(self, source_file, intersection_type, train_target, val_target, temp_dir):
        """Sample using hybrid log-linear binning strategy"""
        print(f"      Applying HYBRID volume distribution sampling")
        print(f"        {self.linear_fraction*100:.0f}% linear, {100-self.linear_fraction*100:.0f}% log bins")
        
        # Get volume binning config
        volume_binning_config = self.config.get("volume_binning", {})
        volume_range = volume_binning_config.get("volume_range")
        
        if not volume_range:
            print(f"        No volume_range specified, falling back to regular sampling")
            regular_sampler = RegularSampler(self.file_processor, self.normalizer)
            return regular_sampler.sample(source_file, intersection_type, train_target, val_target, temp_dir)
        
        config_min_raw = float(volume_range[0])
        config_max_raw = float(volume_range[1])
        
        print(f"        Volume range: [{config_min_raw:.8e}, {config_max_raw:.8e}]")
        print(f"        Bins: {self.nbins} (hybrid: {self.linear_fraction*100:.0f}% linear)")
        print(f"        Targets: {train_target} train + {val_target} val")
        
        # Create hybrid bins
        hybrid_bins = self._create_hybrid_bins()
        
        train_file = os.path.join(temp_dir, f"{intersection_type}_train.csv")
        val_file = os.path.join(temp_dir, f"{intersection_type}_val.csv")
        
        # Apply hybrid sampling
        train_count, val_count = self._guaranteed_hybrid_volume_sampling(
            source_file, train_file, val_file, train_target, val_target, 
            hybrid_bins, (config_min_raw, config_max_raw)
        )
        
        if train_count != train_target or val_count != val_target:
            print(f"        WARNING: Did not achieve exact targets: got {train_count}+{val_count}, wanted {train_target}+{val_target}")
        
        return train_file, val_file, train_count, val_count
    
    def _create_hybrid_bins(self):
        """Create hybrid log-linear bins for optimal coverage"""
        min_val, max_val = self.volume_range
        
        # Calculate number of bins for each region
        n_log_bins = int(self.nbins * (1 - self.linear_fraction))
        n_linear_bins = self.nbins - n_log_bins
        
        # Find transition point in log space
        log_min, log_max = np.log10(min_val), np.log10(max_val)
        transition_log = log_min + (log_max - log_min) * (1 - self.linear_fraction)
        transition_val = 10 ** transition_log
        
        # Create log-spaced bins for small values [1e-8, transition_val]
        if n_log_bins > 0:
            log_bins = np.logspace(np.log10(min_val), np.log10(transition_val), n_log_bins + 1)
            log_bins = log_bins[:-1]  # Remove last point to avoid overlap
        else:
            log_bins = np.array([])
        
        # Create linear-spaced bins for large values [transition_val, 0.02]
        if n_linear_bins > 0:
            linear_bins = np.linspace(transition_val, max_val, n_linear_bins + 1)
        else:
            linear_bins = np.array([max_val])
        
        # Combine bins
        hybrid_bins = np.concatenate([log_bins, linear_bins])
        
        print(f"        Hybrid bins: {n_log_bins} log + {n_linear_bins} linear")
        print(f"        Transition at: {transition_val:.2e}")
        print(f"        Log region: [{min_val:.1e}, {transition_val:.1e}]")
        print(f"        Linear region: [{transition_val:.1e}, {max_val:.1e}]")
        
        return hybrid_bins
    
    def _guaranteed_hybrid_volume_sampling(self, source_file, train_file, val_file, 
                                         train_target, val_target, bin_edges, volume_range):
        """Guaranteed hybrid volume sampling with exact sample counts"""
        
        total_target = train_target + val_target
        n_bins = len(bin_edges) - 1
        samples_per_bin = total_target // n_bins
        remainder = total_target % n_bins
        train_ratio = train_target / total_target if total_target > 0 else 0.9
        
        # Calculate exact targets per bin
        bin_targets = []
        for i in range(n_bins):
            total_in_bin = samples_per_bin + (1 if i < remainder else 0)
            train_in_bin = int(total_in_bin * train_ratio)
            val_in_bin = total_in_bin - train_in_bin
            bin_targets.append((train_in_bin, val_in_bin))
        
        print(f"        Exact targets per hybrid bin:")
        for i, (train_count, val_count) in enumerate(bin_targets):
            print(f"          Bin {i+1:2d}: {train_count} train + {val_count} val = {train_count + val_count} total")
        
        # Phase 1: Collect samples from each bin
        train_bin_samples = [[] for _ in range(n_bins)]
        val_bin_samples = [[] for _ in range(n_bins)]
        
        chunk_size = 10_000
        config_min, config_max = volume_range
        
        print(f"        Phase 1: Collecting samples for hybrid distribution...")
        
        try:
            for chunk_df in pd.read_csv(source_file, chunksize=chunk_size):
                if 'HasIntersection' not in chunk_df.columns or 'IntersectionVolume' not in chunk_df.columns:
                    continue
                
                # Filter for intersecting samples in volume range
                intersecting_mask = chunk_df['HasIntersection'] == 1
                volume_mask = (chunk_df['IntersectionVolume'] >= config_min) & (chunk_df['IntersectionVolume'] <= config_max)
                valid_mask = intersecting_mask & volume_mask
                
                valid_df = chunk_df[valid_mask].copy()
                if len(valid_df) == 0:
                    continue
                
                # Shuffle samples
                valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
                # Assign to hybrid bins
                for _, row in valid_df.iterrows():
                    volume = row['IntersectionVolume']
                    
                    # Find bin (ensure last bin includes max value)
                    bin_idx = np.searchsorted(bin_edges[1:], volume, side='right')
                    bin_idx = min(bin_idx, n_bins - 1)
                    
                    train_target_bin, val_target_bin = bin_targets[bin_idx]
                    sample_data = row.to_dict()
                    
                    # Add to train if needed
                    if len(train_bin_samples[bin_idx]) < train_target_bin:
                        train_bin_samples[bin_idx].append(sample_data)
                    # Add to val if needed
                    elif len(val_bin_samples[bin_idx]) < val_target_bin:
                        val_bin_samples[bin_idx].append(sample_data)
                
                # Check if all bins are full
                all_full = all(
                    len(train_bin_samples[i]) >= bin_targets[i][0] and 
                    len(val_bin_samples[i]) >= bin_targets[i][1]
                    for i in range(n_bins)
                )
                
                if all_full:
                    print(f"        All hybrid bins filled successfully")
                    break
                
                del chunk_df, valid_df
                gc.collect()
        
        except Exception as e:
            print(f"        Error during hybrid sampling: {e}")
            return 0, 0
        
        # Phase 2: Handle unfilled bins with oversampling
        print(f"        Phase 2: Ensuring all hybrid bins are filled...")
        
        for bin_idx in range(n_bins):
            train_target_bin, val_target_bin = bin_targets[bin_idx]
            train_current = len(train_bin_samples[bin_idx])
            val_current = len(val_bin_samples[bin_idx])
            
            train_needed = train_target_bin - train_current
            val_needed = val_target_bin - val_current
            
            if train_needed > 0 or val_needed > 0:
                bin_range = f"[{bin_edges[bin_idx]:.2e}, {bin_edges[bin_idx+1]:.2e}]"
                print(f"          Bin {bin_idx+1} {bin_range} needs {train_needed} train + {val_needed} val samples")
                
                # Collect samples from all bins for oversampling
                all_available_samples = []
                for i in range(n_bins):
                    all_available_samples.extend(train_bin_samples[i])
                    all_available_samples.extend(val_bin_samples[i])
                
                if len(all_available_samples) > 0:
                    # Oversample to fill the bin
                    needed_total = train_needed + val_needed
                    
                    if len(all_available_samples) >= needed_total:
                        selected_indices = np.random.choice(len(all_available_samples), needed_total, replace=False)
                    else:
                        selected_indices = np.random.choice(len(all_available_samples), needed_total, replace=True)
                    
                    selected_samples = [all_available_samples[i] for i in selected_indices]
                    
                    # Modify volumes to fit the target bin
                    bin_min, bin_max = bin_edges[bin_idx], bin_edges[bin_idx+1]
                    for sample in selected_samples:
                        # Scale volume to fit within target bin
                        new_volume = np.random.uniform(bin_min, bin_max)
                        sample['IntersectionVolume'] = new_volume
                    
                    # Add to train/val
                    for i, sample in enumerate(selected_samples):
                        if i < train_needed:
                            train_bin_samples[bin_idx].append(sample)
                        else:
                            val_bin_samples[bin_idx].append(sample)
                    
                    print(f"            Added {len(selected_samples)} hybrid-adjusted samples")
        
        # Phase 3: Write samples with scaling
        print(f"        Phase 3: Writing hybrid samples with scaling...")
        
        train_collected = self._write_samples_with_scaling(train_bin_samples, train_file)
        val_collected = self._write_samples_with_scaling(val_bin_samples, val_file)
        
        print(f"        Final result: {train_collected} train + {val_collected} val (hybrid distribution)")
        
        return train_collected, val_collected
    
    def _write_samples_with_scaling(self, bin_samples, output_file):
        """Write samples to file WITH scaling applied"""
        all_samples = []
        for bin_sample_list in bin_samples:
            all_samples.extend(bin_sample_list)
        
        if len(all_samples) == 0:
            pd.DataFrame().to_csv(output_file, index=False)
            return 0
        
        # Convert to DataFrame and apply scaling
        samples_df = pd.DataFrame(all_samples)
        samples_df = self.normalizer.normalize_data_types_with_scaling(samples_df)
        samples_df.to_csv(output_file, index=False, float_format='%.17g')
        
        print(f"        Saved {len(all_samples)} hybrid samples to {os.path.basename(output_file)}")
        return len(all_samples)

class SamplingStrategySelector:
    def __init__(self, config, file_processor, normalizer):
        self.config = config
        self.file_processor = file_processor
        self.normalizer = normalizer
    
    def get_strategy(self, intersection_type):
        if intersection_type == "polyhedron_intersection":
            sampling_strategy = self.config.get("sampling_strategy", "log_uniform_volume")
            
            if sampling_strategy == "log_uniform_volume":
                return LogUniformVolumeSampler(self.config, self.normalizer, self.file_processor)
            elif sampling_strategy == "hybrid_volume":  # NEW
                return HybridVolumeSampler(self.config, self.normalizer, self.file_processor)
            else:
                return UniformVolumeSampler(self.config, self.normalizer, self.file_processor)
        else:
            return RegularSampler(self.file_processor, self.normalizer)


# ============================================================================
# 4. DATA ENHANCEMENT LAYER
# ============================================================================


class AugmentationEngine:
    @staticmethod
    def augment_data(data: pd.DataFrame, config) -> pd.DataFrame:
        if data.empty:
            return data
            
        try:
            augmentation_config = config.get("augmentations", {})
            if not augmentation_config:
                return data
            
            print(f"  Starting augmentation on {len(data)} samples...")
            
            # Apply geometric augmentations first (these create new samples)
            data = AugmentationEngine._apply_geometric_augmentations(data, augmentation_config)
            
            # Then apply sorting augmentations (these reorder existing samples)
            data = AugmentationEngine._apply_sorting_augmentations(data, augmentation_config)
                
        except Exception as e:
            print(f"Warning: Data augmentation failed: {e}")
            import traceback
            traceback.print_exc()
            
        return data
    
    @staticmethod
    def _apply_geometric_augmentations(data: pd.DataFrame, augmentation_config) -> pd.DataFrame:
        """Apply geometric augmentations that create new samples based on percentages"""
        
        # Get augmentation percentages
        point_wise_pct = augmentation_config.get("point_wise_permutation_augmentation_pct", 0)
        tetrahedron_wise_pct = augmentation_config.get("tetrahedron_wise_permutation_augmentation_pct", 0)
        
        if point_wise_pct == 0 and tetrahedron_wise_pct == 0:
            print(f"    No geometric augmentations configured")
            return data
        
        original_count = len(data)
        augmented_samples = []
        
        # Point-wise permutation augmentation
        if point_wise_pct > 0:
            n_point_aug = int(original_count * (point_wise_pct / 100.0))
            if n_point_aug > 0:
                print(f"    Applying point-wise permutation to {n_point_aug} samples ({point_wise_pct}%)")
                point_aug_samples = AugmentationEngine._create_point_wise_augmented_samples(
                    data, n_point_aug
                )
                augmented_samples.extend(point_aug_samples)
        
        # Tetrahedron-wise permutation augmentation
        if tetrahedron_wise_pct > 0:
            n_tetra_aug = int(original_count * (tetrahedron_wise_pct / 100.0))
            if n_tetra_aug > 0:
                print(f"    Applying tetrahedron-wise permutation to {n_tetra_aug} samples ({tetrahedron_wise_pct}%)")
                tetra_aug_samples = AugmentationEngine._create_tetrahedron_wise_augmented_samples(
                    data, n_tetra_aug
                )
                augmented_samples.extend(tetra_aug_samples)
        
        # Combine original data with augmented samples
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            combined_data = pd.concat([data, augmented_df], ignore_index=True)
            print(f"    Geometric augmentation complete: {original_count} → {len(combined_data)} samples")
            return combined_data
        else:
            return data
    
    @staticmethod
    def _create_point_wise_augmented_samples(data: pd.DataFrame, n_samples: int) -> list:
        """Create augmented samples using point-wise permutation within tetrahedra"""
        augmented_samples = []
        
        # Randomly select samples to augment
        if n_samples >= len(data):
            selected_indices = np.random.choice(len(data), n_samples, replace=True)
        else:
            selected_indices = np.random.choice(len(data), n_samples, replace=False)
        
        coordinate_columns = data.columns[:-2]
        
        for idx in selected_indices:
            original_sample = data.iloc[idx].copy()
            
            try:
                # Apply point-wise permutation using GeometryUtils
                coords_tensor = torch.tensor(
                    original_sample[coordinate_columns].values, 
                    dtype=torch.float64
                ).unsqueeze(0)  # Add batch dimension
                
                augmented_coords = gu.permute_points_within_tetrahedrons(coords_tensor)
                augmented_coords_flat = augmented_coords.squeeze(0).numpy()
                
                # Create augmented sample
                augmented_sample = original_sample.copy()
                augmented_sample[coordinate_columns] = augmented_coords_flat
                
                augmented_samples.append(augmented_sample.to_dict())
                
            except Exception as e:
                print(f"      Warning: Point-wise augmentation failed for sample {idx}: {e}")
                # Keep original sample if augmentation fails
                augmented_samples.append(original_sample.to_dict())
        
        return augmented_samples
    
    @staticmethod
    def _create_tetrahedron_wise_augmented_samples(data: pd.DataFrame, n_samples: int) -> list:
        """Create augmented samples using tetrahedron swapping"""
        augmented_samples = []
        
        # Randomly select samples to augment
        if n_samples >= len(data):
            selected_indices = np.random.choice(len(data), n_samples, replace=True)
        else:
            selected_indices = np.random.choice(len(data), n_samples, replace=False)
        
        coordinate_columns = data.columns[:-2]
        
        for idx in selected_indices:
            original_sample = data.iloc[idx].copy()
            
            try:
                # Apply tetrahedron swapping using GeometryUtils
                coords_tensor = torch.tensor(
                    original_sample[coordinate_columns].values, 
                    dtype=torch.float64
                ).unsqueeze(0)  # Add batch dimension
                
                augmented_coords = gu.swap_tetrahedrons(coords_tensor)
                augmented_coords_flat = augmented_coords.squeeze(0).numpy()
                
                # Create augmented sample
                augmented_sample = original_sample.copy()
                augmented_sample[coordinate_columns] = augmented_coords_flat
                
                augmented_samples.append(augmented_sample.to_dict())
                
            except Exception as e:
                print(f"      Warning: Tetrahedron-wise augmentation failed for sample {idx}: {e}")
                # Keep original sample if augmentation fails
                augmented_samples.append(original_sample.to_dict())
        
        return augmented_samples
    
    @staticmethod
    def _apply_sorting_augmentations(data: pd.DataFrame, augmentation_config) -> pd.DataFrame:
        """Apply sorting-based augmentations using GeometryUtils functions"""
        
        # Handle y_sorting (difficulty-based)
        y_sorting_config = augmentation_config.get("y_sorting", {})
        if y_sorting_config:
            difficulty_based_config = y_sorting_config.get("difficulty_based")
            if difficulty_based_config:
                print(f"    Applying difficulty-based sorting: {difficulty_based_config}")
                
                if difficulty_based_config == "easier_first":
                    data = gu.sort_by_difficulty(data, easier_first=True)
                elif difficulty_based_config == "harder_first":
                    data = gu.sort_by_difficulty(data, easier_first=False)
                elif difficulty_based_config == "stepwise_easier":
                    data = gu.sort_by_difficulty_stepwise(data, easier_first=True)
                elif difficulty_based_config == "stepwise_harder":
                    data = gu.sort_by_difficulty_stepwise(data, easier_first=False)
                elif difficulty_based_config == "mixed_easy":
                    data = gu.sort_by_difficulty_mixed(data, easier_first=True, mix_ratio=0.3)
                elif difficulty_based_config == "mixed_hard":
                    data = gu.sort_by_difficulty_mixed(data, easier_first=False, mix_ratio=0.3)
                else:
                    print(f"      Warning: Unknown difficulty config '{difficulty_based_config}'")
                
                print(f"      Applied difficulty-based sorting: {difficulty_based_config}")
                    
        # Handle x_sorting (spatial and volume-based)
        x_sorting_config = augmentation_config.get("x_sorting", {})
        if x_sorting_config:
            
            # Volume-based sorting using GeometryUtils
            volume_based_config = x_sorting_config.get("volume_based")
            if volume_based_config:
                print(f"    Applying volume-based sorting: {volume_based_config}")
                
                if volume_based_config == "bigger":
                    data = gu.volume_reordering(data, larger=True)
                elif volume_based_config == "smaller":
                    data = gu.volume_reordering(data, larger=False)
                else:
                    print(f"      Warning: Unknown volume_based config '{volume_based_config}', expected 'bigger' or 'smaller'")
                    # Default to bigger first
                    data = gu.volume_reordering(data, larger=True)
                
                print(f"      Applied volume-based sorting: {volume_based_config}")
                    
            # Spatial-based sorting using GeometryUtils
            spatial_based_config = x_sorting_config.get("spatial_based")
            if spatial_based_config:
                print(f"    Applying spatial-based sorting: {spatial_based_config}")
                
                # Handle both string and dict configurations
                if isinstance(spatial_based_config, str):
                    if spatial_based_config == "x_coordinate":
                        data = gu.sort_by_x_coordinate(data, column_name="T1_v1_x")
                        print(f"      Sorted by T1_v1_x coordinate")
                    elif spatial_based_config == "x_coordinate_alt":
                        data = gu.sort_by_x_coordinate_alt(data)
                        print(f"      Applied alternative x-coordinate sorting")
                    elif spatial_based_config == "morton_code":
                        data = gu.sort_by_morton_code(data)
                        print(f"      Applied Morton code spatial sorting")
                    elif spatial_based_config == "morton_code_alt":
                        data = gu.sort_by_morton_code_alt(data)
                        print(f"      Applied alternative Morton code sorting")
                    else:
                        # Default to x-coordinate sorting
                        data = gu.sort_by_x_coordinate(data, column_name="T1_v1_x")
                        print(f"      Applied default x-coordinate sorting")
                        
                elif isinstance(spatial_based_config, dict):
                    # Dictionary configuration with options
                    sort_column = spatial_based_config.get("column", "T1_v1_x")
                    ascending = spatial_based_config.get("ascending", True)
                    sort_type = spatial_based_config.get("type", "x_coordinate")
                    
                    if sort_type == "x_coordinate":
                        data = gu.sort_by_x_coordinate(data, column_name=sort_column)
                        print(f"      Sorted by column '{sort_column}' (ascending={ascending})")
                    elif sort_type == "morton_code":
                        scale_factor = spatial_based_config.get("scale_factor", 1e18)
                        data = gu.sort_by_morton_code_alt(data, scale_factor=scale_factor)
                        print(f"      Applied Morton code sorting (scale={scale_factor})")
                    else:
                        # Fallback to x-coordinate
                        data = gu.sort_by_x_coordinate(data, column_name=sort_column)
                        print(f"      Applied fallback x-coordinate sorting")
                        
                else:
                    raise ValueError(f"Invalid spatial_based configuration: {spatial_based_config}")
            
            # Intersection volume sorting using GeometryUtils
            intersection_volume_config = x_sorting_config.get("intersection_volume_based")
            if intersection_volume_config:
                print(f"    Applying intersection volume sorting")
                data = gu.sort_by_intersection_volume_whole_dataset(data)
                print(f"      Applied intersection volume sorting")
        
        else:
            print(f"    No x_sorting configuration found, skipping spatial/volume sorting")
        
        return data


class TransformationEngine:

    @staticmethod
    def transform_data(data: pd.DataFrame, config) -> pd.DataFrame:
        if data.empty:
            return data
        
        transformation_config = config.get("transformations")
        if not transformation_config:
            return data
        
        # Normalize transformation list
        transformation_list = (
            [transformation_config] if isinstance(transformation_config, str) and transformation_config.strip()
            else transformation_config if isinstance(transformation_config, list)
            else []
        )
        
        if not transformation_list:
            return data
        
        try:
            coordinate_columns = data.columns[:-2]
            print(f"  Validating {len(data)} rows for data integrity...")
            
            # MEMORY EFFICIENT: Single pass validation using vectorized operations
            # Convert all coordinate columns at once and find invalid samples
            numeric_data = data[coordinate_columns].apply(pd.to_numeric, errors='coerce')
            invalid_mask = numeric_data.isna().any(axis=1)
            
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                print(f"    Removing {invalid_count} samples with invalid coordinates")
                # Single boolean indexing operation - no intermediate copies
                data = data[~invalid_mask].reset_index(drop=True)
                print(f"    Dataset: {len(data) + invalid_count} → {len(data)} samples")
            
            # MEMORY EFFICIENT: In-place update of coordinate columns
            data[coordinate_columns] = numeric_data[~invalid_mask].reset_index(drop=True) if invalid_count > 0 else numeric_data
            
            # Apply transformations
            for transformation_name in transformation_list:
                print(f"  Applying {transformation_name}...")
                data = TransformationEngine._apply_transformation(data, transformation_name, coordinate_columns)
                # Update coordinate_columns after transformation (dimension may change)
                coordinate_columns = data.columns[:-2]
            
            return data
            
        except Exception as e:
            print(f"Warning: Transformation failed: {e}")
            return data

    @staticmethod
    def _apply_transformation(data: pd.DataFrame, transformation_name: str, coordinate_columns) -> pd.DataFrame:
        """Apply single transformation - dispatches to appropriate method"""
        transformation_map = {
            "unitary_tetrahedron": TransformationEngine._apply_unitary_vectorized,
            "principal_axis": TransformationEngine._apply_principal_vectorized
        }
        
        transform_func = transformation_map.get(transformation_name)
        if not transform_func:
            print(f"    Unknown transformation: {transformation_name}")
            return data
        
        return transform_func(data, coordinate_columns)

    @staticmethod
    def _apply_unitary_vectorized(data: pd.DataFrame, coordinate_columns) -> pd.DataFrame:
        """Apply unitary transformation with fallback - handles 24->12 dimension change"""
        try:
            # MEMORY EFFICIENT: Direct tensor conversion from DataFrame values
            data_tensor = torch.tensor(data[coordinate_columns].values, dtype=torch.float64)
            transformed_tensor = gu.apply_unitary_tetrahedron_transformation_batch(data_tensor)
            
            # Check what we actually got back
            actual_output_cols = transformed_tensor.shape[1]
            
            print(f"    Debug: Input shape: {data_tensor.shape}, Output shape: {transformed_tensor.shape}")
            
            if actual_output_cols == 12:
                # Expected case: 24 -> 12 transformation
                new_columns = []
                for v_idx in range(4):
                    for coord in ['x', 'y', 'z']:
                        new_columns.append(f'T1_v{v_idx+1}_{coord}')
                        
            elif actual_output_cols == 24:
                # Unexpected case: transformation didn't reduce dimensions
                print(f"    Warning: Transformation returned 24 columns instead of 12")
                # Check if this is actually the correct unitary transformation
                # For now, use the first 12 columns (assuming they represent the transformed tetrahedron)
                transformed_tensor = transformed_tensor[:, :12]
                actual_output_cols = 12
                
                new_columns = []
                for v_idx in range(4):
                    for coord in ['x', 'y', 'z']:
                        new_columns.append(f'T1_v{v_idx+1}_{coord}')
                        
            else:
                raise ValueError(f"Unexpected output dimension: {actual_output_cols}")
            
            # Update DataFrame with transformed data
            # Remove original coordinate columns and add new ones
            data_transformed = data.drop(columns=coordinate_columns).copy()
            
            # Add transformed columns
            transformed_df = pd.DataFrame(
                transformed_tensor.cpu().numpy(), 
                columns=new_columns,
                index=data.index
            )
            
            # Combine with non-coordinate columns
            result = pd.concat([transformed_df, data_transformed], axis=1)
            
            print(f"    ✓ Vectorized processing: {len(data)} samples, {len(coordinate_columns)}→{actual_output_cols} features")
            return result
            
        except Exception as e:
            print(f"    Vectorized failed ({e}), using fallback")
            return TransformationEngine._apply_unitary_fallback(data, coordinate_columns)

    @staticmethod
    def _apply_unitary_fallback(data: pd.DataFrame, coordinate_columns) -> pd.DataFrame:
        """Memory-efficient row-by-row fallback for unitary transformation"""
        failed_rows = 0
        transformed_rows = []
        
        print(f"    Using fallback transformation for {len(data)} rows")
        
        for idx in data.index:
            try:
                row = data.loc[idx, coordinate_columns]
                transformed_row = gu.apply_unitary_tetrahedron_transformation(row)
                
                # Check the output dimensions
                if hasattr(transformed_row, 'values'):
                    transformed_values = transformed_row.values
                else:
                    transformed_values = transformed_row
                    
                if len(transformed_values) == 12:
                    transformed_rows.append(transformed_values)
                elif len(transformed_values) == 24:
                    # Take first 12 if we got 24
                    print(f"    Warning: Single row transformation returned 24 values, taking first 12")
                    transformed_rows.append(transformed_values[:12])
                else:
                    print(f"    Warning: Unexpected transformation output size: {len(transformed_values)}")
                    # Create zero vector for unexpected output
                    transformed_rows.append(np.zeros(12))
                    failed_rows += 1
                    
            except Exception as e:
                print(f"    Warning: Row {idx} transformation failed: {e}")
                failed_rows += 1
                # Create zero vector for failed transformation
                transformed_rows.append(np.zeros(12))
        
        # Create new DataFrame with transformed features
        new_columns = []
        for v_idx in range(4):
            for coord in ['x', 'y', 'z']:
                new_columns.append(f'T1_v{v_idx+1}_{coord}')
        
        try:
            transformed_df = pd.DataFrame(
                transformed_rows, 
                columns=new_columns,
                index=data.index
            )
            
            # Combine with non-coordinate columns
            non_coordinate_data = data.drop(columns=coordinate_columns)
            result = pd.concat([transformed_df, non_coordinate_data], axis=1)
            
            if failed_rows > 0:
                print(f"    ⚠ {failed_rows} rows failed transformation (kept as zeros)")
            
            print(f"    ✓ Fallback completed: {len(data)} samples, 24→12 features")
            return result
            
        except Exception as e:
            print(f"    Error in fallback DataFrame creation: {e}")
            print(f"    Transformed rows shape: {len(transformed_rows)} x {len(transformed_rows[0]) if transformed_rows else 0}")
            print(f"    New columns: {len(new_columns)}")
            raise

    @staticmethod
    def _apply_principal_vectorized(data: pd.DataFrame, coordinate_columns) -> pd.DataFrame:
        """Apply principal axis transformation with fallback - maintains 24 features"""
        try:
            # MEMORY EFFICIENT: Direct tensor conversion from DataFrame values
            data_tensor = torch.tensor(data[coordinate_columns].values, dtype=torch.float64)
            transformed_tensor = gu.apply_principal_axis_transformation_batch(data_tensor)
            
            # IN-PLACE update of coordinate columns (maintains same dimensions)
            data[coordinate_columns] = transformed_tensor.cpu().numpy()
            print(f"    ✓ Vectorized processing: {len(data)} samples")
            return data
            
        except Exception as e:
            print(f"    Vectorized failed ({e}), using fallback")
            return TransformationEngine._apply_principal_fallback(data, coordinate_columns)

    @staticmethod
    def _apply_principal_fallback(data: pd.DataFrame, coordinate_columns) -> pd.DataFrame:
        """Memory-efficient row-by-row fallback for principal axis transformation"""
        failed_rows = 0
        for idx in data.index:
            try:
                row_tensor = torch.tensor(data.loc[idx, coordinate_columns].values, dtype=torch.float64)
                transformed = gu.apply_principal_axis_transformation(row_tensor)
                data.loc[idx, coordinate_columns] = transformed.cpu().numpy().flatten()
            except Exception:
                failed_rows += 1
        
        if failed_rows > 0:
            print(f"    ⚠ {failed_rows} rows failed transformation (kept original)")
        return data

# ============================================================================
# 5. OUTPUT MANAGEMENT LAYER
# ============================================================================

class DistributionReporter:
    def __init__(self, file_processor, normalizer):
        self.file_processor = file_processor
        self.normalizer = normalizer
    
    def check_data_distribution(self, file_path, dataset_name):
        try:
            sample_size = min(5000, self.file_processor.count_samples_in_file(file_path))
            sample_df = pd.read_csv(file_path, nrows=sample_size)
            
            if len(sample_df) == 0:
                print(f"{dataset_name} dataset is empty!")
                return
            
            print(f"{dataset_name} distribution (first {sample_size} samples):")
            
            intersection_counts = sample_df['HasIntersection'].value_counts()
            total_samples = len(sample_df)
            
            for label, count in intersection_counts.items():
                percentage = (count / total_samples) * 100
                if label == 0:
                    print(f"  No intersection: {count} samples ({percentage:.1f}%)")
                else:
                    print(f"  Has intersection: {count} samples ({percentage:.1f}%)")
            
            intersecting_samples = sample_df[sample_df['HasIntersection'] == 1]
            if len(intersecting_samples) > 0:
                volumes = intersecting_samples['IntersectionVolume']
                vol_min = volumes.min()
                vol_max = volumes.max()
                vol_mean = volumes.mean()
                vol_std = volumes.std()
                print(f"  Intersection volumes:")
                print(f"    min={vol_min:.8e}, max={vol_max:.8e}")
                print(f"    mean={vol_mean:.8e}, std={vol_std:.8e}")
                
                volume_positive = intersecting_samples[intersecting_samples['IntersectionVolume'] > 0]
                if len(volume_positive) > 0:
                    print(f"    Samples with volume > 0: {len(volume_positive)}")
                    
        except Exception as e:
            print(f"Could not check {dataset_name} distribution: {e}")

class OutputCoordinator:
    def __init__(self, file_processor, reporter):
        self.file_processor = file_processor
        self.reporter = reporter
    
    def save_processed_data(self, train_files, val_files, processed_data_path):
        os.makedirs(os.path.join(processed_data_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(processed_data_path, "val"), exist_ok=True)
        
        train_file_path = os.path.join(processed_data_path, "train", "train_data.csv")
        val_file_path = os.path.join(processed_data_path, "val", "val_data.csv")
        
        if train_files:
            print(f"Combining {len(train_files)} train files...")
            expected_train_count = self.file_processor.count_samples_in_files(train_files)
            print(f"  Expected train samples: {expected_train_count}")
            
            self.file_processor.combine_files_streaming(train_files, train_file_path)
            self.file_processor.validate_and_fix_file_consistency(train_file_path)
            
            final_count = self.file_processor.count_samples_in_file(train_file_path)
            print(f"Training data saved: {final_count} samples")
            
            self.reporter.check_data_distribution(train_file_path, "Training")
        
        if val_files:
            print(f"Combining {len(val_files)} val files...")
            expected_val_count = self.file_processor.count_samples_in_files(val_files)
            print(f"  Expected val samples: {expected_val_count}")
            
            self.file_processor.combine_files_streaming(val_files, val_file_path)
            self.file_processor.validate_and_fix_file_consistency(val_file_path)
            
            final_count = self.file_processor.count_samples_in_file(val_file_path)
            print(f"Validation data saved: {final_count} samples")
            
            self.reporter.check_data_distribution(val_file_path, "Validation")

# ============================================================================
# 6. MAIN PROCESSOR CLASS
# ============================================================================

class CDataProcessor:
    def __init__(self, processor_config):
        if processor_config is None:
            raise ValueError("processor_config cannot be None. Please provide a valid configuration.")
        
        self.config = processor_config
        self.temp_dir = None
        self.train_files = []
        self.val_files = []
        
        # Initialize components
        self.file_processor = FileProcessor()
        self.file_scanner = FileSystemScanner(self.config["dataset_paths"]["raw_data"])
        self.data_inventory = DataInventory(self.file_processor)
        self.normalizer = DataTypeNormalizer(self.config)
        self.strategy_selector = SamplingStrategySelector(self.config, self.file_processor, self.normalizer)
        self.reporter = DistributionReporter(self.file_processor, self.normalizer)
        self.output_coordinator = OutputCoordinator(self.file_processor, self.reporter)
        
        self._validate_config()
    
    def _validate_config(self):
        min_volume_threshold = self.config.get("min_volume_threshold", 1e-8)
        
        if isinstance(min_volume_threshold, str):
            try:
                min_volume_threshold = float(min_volume_threshold)
                self.config["min_volume_threshold"] = min_volume_threshold
            except ValueError:
                raise ValueError(f"min_volume_threshold must be a valid number, got '{min_volume_threshold}'")
        else:
            min_volume_threshold = float(min_volume_threshold)
            self.config["min_volume_threshold"] = min_volume_threshold
        
        if min_volume_threshold <= 0:
            raise ValueError(f"min_volume_threshold must be positive, got {min_volume_threshold}")
        if min_volume_threshold > 0.2:
            print(f"Warning: min_volume_threshold {min_volume_threshold} is very large")
    
    def process(self):
        print("-- Processing Data --")
        
        self.temp_dir = tempfile.mkdtemp(prefix="data_processor_")
        print(f"Using temporary directory: {self.temp_dir}")
        
        try:
            # Stage 1: Discovery & Planning
            self._discovery_and_planning_stage()
            
            # Stage 2: Quality Processing & Strategic Sampling
            self._quality_processing_and_sampling_stage()
            
            # Stage 3: Enhancement Pipeline
            self._enhancement_pipeline_stage()
            
            # Stage 4: Finalization
            self._finalization_stage()
            
        finally:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print("Cleaned up temporary files")
        
        print("-- Data Processing Complete --")
    
    def _discovery_and_planning_stage(self):
        print("\n=== Stage 1: Discovery & Planning ===")
        
        total_raw = self.data_inventory.count_raw_data(self.config["dataset_paths"]["raw_data"])
        total_target = self.config["num_train_samples"] + self.config["num_val_samples"]

        if total_raw < total_target:
            print(f"WARNING: Raw data ({total_raw:,}) insufficient for target ({total_target:,})")
        
        train_samples = self.config["num_train_samples"]
        val_samples = self.config["num_val_samples"]
        
        print(f"Target: {train_samples} train, {val_samples} val samples")

    def _calculate_exact_sample_allocation(self, total_train_target, total_val_target, intersection_distributions, intersection_files):
        """Calculate exact sample allocation ensuring percentages are met precisely"""
        
        print(f"\n  Calculating exact sample allocation:")
        print(f"    Total targets: {total_train_target} train + {total_val_target} val")
        
        # Verify percentages sum to 100
        total_percentage = sum(intersection_distributions.values())
        if abs(total_percentage - 100.0) > 0.001:
            raise ValueError(f"Intersection distributions must sum to 100%, got {total_percentage}%")
        
        train_targets = {}
        val_targets = {}
        
        # Calculate exact samples for each intersection type
        allocated_train = 0
        allocated_val = 0
        
        # Sort by percentage (largest first) to handle remainders better
        sorted_types = sorted(intersection_files.keys(), 
                             key=lambda x: intersection_distributions.get(x, 0), 
                             reverse=True)
        
        for i, intersection_type in enumerate(sorted_types):
            if intersection_type not in intersection_distributions:
                train_targets[intersection_type] = 0
                val_targets[intersection_type] = 0
                continue
                
            percentage = intersection_distributions[intersection_type] / 100.0
            
            if i == len(sorted_types) - 1:
                # Last type gets all remaining samples to ensure exact total
                train_targets[intersection_type] = total_train_target - allocated_train
                val_targets[intersection_type] = total_val_target - allocated_val
            else:
                # Use floor for intermediate calculations
                train_count = int(total_train_target * percentage)
                val_count = int(total_val_target * percentage)
                
                train_targets[intersection_type] = train_count
                val_targets[intersection_type] = val_count
                
                allocated_train += train_count
                allocated_val += val_count
        
        # Verify exact totals
        final_train_total = sum(train_targets.values())
        final_val_total = sum(val_targets.values())
        
        print(f"    Exact allocation by type:")
        for intersection_type in intersection_files.keys():
            train_count = train_targets[intersection_type]
            val_count = val_targets[intersection_type]
            total_count = train_count + val_count
            actual_percentage = (total_count / (total_train_target + total_val_target)) * 100
            expected_percentage = intersection_distributions.get(intersection_type, 0)
            
            print(f"      {intersection_type}: {train_count} train + {val_count} val = {total_count} total")
            print(f"        Expected: {expected_percentage}%, Actual: {actual_percentage:.2f}%")
        
        print(f"    Final totals: {final_train_total} train + {final_val_total} val")
        
        if final_train_total != total_train_target or final_val_total != total_val_target:
            raise ValueError(f"Sample allocation error: got {final_train_total}+{final_val_total}, expected {total_train_target}+{total_val_target}")
        
        return train_targets, val_targets

    def _enforce_exact_sample_counts(self, intersection_files, train_targets, val_targets):
        """Process each intersection type and guarantee exact sample counts"""
        
        print(f"\n  Processing with guaranteed sample counts:")
        
        all_train_files = []
        all_val_files = []
        
        for intersection_type, source_file in intersection_files.items():
            current_train_target = train_targets[intersection_type]
            current_val_target = val_targets[intersection_type]
            
            if current_train_target == 0 and current_val_target == 0:
                print(f"\n    Skipping {intersection_type} (0 samples allocated)")
                continue
                
            print(f"\n    Processing {intersection_type}...")
            print(f"      Required: {current_train_target} train + {current_val_target} val (EXACT)")
            
            # Attempt initial sampling
            try:
                train_file, val_file, train_count, val_count = self._process_intersection_type(
                    intersection_type, source_file, current_train_target, current_val_target
                )
                
                print(f"      Initial result: {train_count} train + {val_count} val")
                
                # Adjust to exact counts if needed
                if train_file and train_count != current_train_target:
                    print(f"      Adjusting train samples: {train_count} → {current_train_target}")
                    train_file = self._adjust_file_to_exact_count(
                        train_file, current_train_target, intersection_type, "train"
                    )
                    train_count = current_train_target
                
                if val_file and val_count != current_val_target:
                    print(f"      Adjusting val samples: {val_count} → {current_val_target}")
                    val_file = self._adjust_file_to_exact_count(
                        val_file, current_val_target, intersection_type, "val"
                    )
                    val_count = current_val_target
                
                if train_file and os.path.exists(train_file):
                    all_train_files.append(train_file)
                
                if val_file and os.path.exists(val_file):
                    all_val_files.append(val_file)
                
                print(f"      ✓ Final: {train_count} train + {val_count} val (GUARANTEED)")
                
            except Exception as e:
                print(f"      ✗ Error processing {intersection_type}: {e}")
                raise
        
        return all_train_files, all_val_files

    def _adjust_file_to_exact_count(self, source_file, target_count, intersection_type, split):
        """Adjust a file to contain exactly the target number of samples"""
        current_count = self.file_processor.count_samples_in_file(source_file)
        
        if current_count == target_count:
            return source_file
        
        adjusted_file = os.path.join(self.temp_dir, f"{intersection_type}_{split}_exact.csv")
        
        if current_count > target_count:
            # Truncate to exact count
            print(f"        Truncating from {current_count} to {target_count}")
            chunk_size = min(target_count, 50000)
            samples_written = 0
            first_chunk = True
            
            for chunk_df in pd.read_csv(source_file, chunksize=chunk_size):
                if samples_written >= target_count:
                    break
                
                remaining_needed = target_count - samples_written
                if len(chunk_df) > remaining_needed:
                    chunk_df = chunk_df.iloc[:remaining_needed]
                
                chunk_df.to_csv(adjusted_file, mode='a', header=first_chunk, index=False, float_format='%.17g')
                first_chunk = False
                samples_written += len(chunk_df)
                
                del chunk_df
                gc.collect()
                
        else:
            # Oversample to reach exact count
            print(f"        Oversampling from {current_count} to {target_count}")
            
            # Read all current data
            all_data = pd.read_csv(source_file)
            needed_extra = target_count - current_count
            
            if len(all_data) > 0:
                # Create oversampled data
                if needed_extra <= len(all_data):
                    # Sample without replacement
                    extra_samples = all_data.sample(n=needed_extra, replace=False, random_state=42)
                else:
                    # Sample with replacement
                    extra_samples = all_data.sample(n=needed_extra, replace=True, random_state=42)
                
                # Combine and save
                combined_data = pd.concat([all_data, extra_samples], ignore_index=True)
                combined_data.to_csv(adjusted_file, index=False, float_format='%.17g')
            else:
                # No data to oversample from - create empty file
                pd.DataFrame().to_csv(adjusted_file, index=False)
        
        # Verify exact count
        final_count = self.file_processor.count_samples_in_file(adjusted_file)
        if final_count != target_count:
            raise ValueError(f"Failed to create exact count file: got {final_count}, wanted {target_count}")
        
        return adjusted_file

    def _quality_processing_and_sampling_stage(self):
        """Quality processing and sampling stage with guaranteed percentages"""
        print("\n=== Stage 2: Quality Processing & Strategic Sampling ===")
        
        intersection_types = [
            'no_intersection',
            'point_intersection', 
            'segment_intersection',
            'polygon_intersection',
            'polyhedron_intersection'
        ]
        
        intersection_files = {}
        
        # Discover and combine files for each intersection type
        for intersection_type in intersection_types:
            files = self.file_scanner.discover_files(intersection_type)
            if files:
                if len(files) == 1:
                    intersection_files[intersection_type] = files[0]
                    print(f"  Found {intersection_type}: {os.path.basename(files[0])}")
                else:
                    print(f"  Found {len(files)} files for {intersection_type}, combining...")
                    combined_file = self._combine_raw_files(files, intersection_type)
                    if combined_file:
                        intersection_files[intersection_type] = combined_file
                        print(f"    Combined into: {os.path.basename(combined_file)}")
            else:
                print(f"  Warning: No files found for {intersection_type}")
        
        if not intersection_files:
            raise FileNotFoundError("No intersection files found for processing.")
        
        print(f"  Total intersection types to process: {len(intersection_files)}")
        
        # Use intersection_distributions from config
        total_train_target = self.config.get("num_train_samples", 10000)
        total_val_target = self.config.get("num_val_samples", 1000)
        
        # Get distribution percentages from config
        intersection_distributions = self.config.get("intersection_distributions", {})
        
        if not intersection_distributions:
            # Fallback to equal distribution
            print(f"  No intersection_distributions found, using equal distribution")
            intersection_distributions = {itype: 100.0/len(intersection_files) 
                                        for itype in intersection_files.keys()}
        
        print(f"  Using configured intersection distributions:")
        for itype, percentage in intersection_distributions.items():
            print(f"    {itype}: {percentage}%")
        
        # Calculate EXACT sample allocation
        train_targets, val_targets = self._calculate_exact_sample_allocation(
            total_train_target, total_val_target, intersection_distributions, intersection_files
        )
        
        # Process with guaranteed sample counts
        all_train_files, all_val_files = self._enforce_exact_sample_counts(
            intersection_files, train_targets, val_targets
        )
        
        # Store the files for enhancement pipeline
        self.train_files = all_train_files
        self.val_files = all_val_files
        
        # Verify final totals
        final_train_total = sum(self.file_processor.count_samples_in_file(f) for f in all_train_files if f)
        final_val_total = sum(self.file_processor.count_samples_in_file(f) for f in all_val_files if f)
        
        print(f"\n  ✓ GUARANTEED Quality Processing Complete:")
        print(f"    Train files: {len(all_train_files)} ({final_train_total} samples)")
        print(f"    Val files: {len(all_val_files)} ({final_val_total} samples)")
        print(f"    Target: {total_train_target} train + {total_val_target} val")
        
        if final_train_total == total_train_target and final_val_total == total_val_target:
            print(f"    ✓ EXACT PERCENTAGES ACHIEVED!")
        else:
            raise ValueError(f"Failed to achieve exact percentages: got {final_train_total}+{final_val_total}")

    def _combine_raw_files(self, file_list, intersection_type):
        """Combine multiple raw files for an intersection type"""
        if not file_list:
            return None
        
        if len(file_list) == 1:
            return file_list[0]
        
        combined_file = os.path.join(self.temp_dir, f"{intersection_type}_combined_raw.csv")
        
        print(f"    Combining {len(file_list)} files:")
        for file_path in file_list:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"      {os.path.basename(file_path)} ({file_size:.1f} MB)")
        
        chunk_size = 50_000
        total_combined = 0
        first_chunk = True
        
        try:
            for file_path in file_list:
                print(f"    Processing {os.path.basename(file_path)}...")
                file_total = 0
                
                for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
                    # Basic type conversion only (no scaling yet)
                    chunk_df = self.normalizer.basic_conversion_only(chunk_df)
                    
                    if len(chunk_df) > 0:
                        chunk_df.to_csv(combined_file, mode='a', header=first_chunk, index=False,
                                    float_format='%.17g')
                        first_chunk = False
                        total_combined += len(chunk_df)
                        file_total += len(chunk_df)
                    
                    del chunk_df
                    gc.collect()
                
                print(f"      Added {file_total} samples")
        
        except Exception as e:
            print(f"    Error combining files: {e}")
            return None
        
        print(f"    Total combined: {total_combined} samples")
        return combined_file if total_combined > 0 else None

    def _process_intersection_type(self, intersection_type, source_file, train_target, val_target):
        """Process a specific intersection type and return all required values"""
        
        # Get the sampling strategy based on intersection type
        strategy = self.strategy_selector.get_strategy(intersection_type)
        
        try:
            # Call the sampling strategy
            result = strategy.sample(source_file, intersection_type, train_target, val_target, self.temp_dir)
            
            # Check what the strategy returned
            if isinstance(result, tuple) and len(result) == 4:
                return result
            else:
                raise ValueError(f"Strategy returned unexpected result: {result}")
                
        except Exception as e:
            print(f"  Error in sampling strategy: {e}")
            raise

    def _enhancement_pipeline_stage(self):
        print("\n=== Stage 3: Enhancement Pipeline ===")
        
        if self._should_apply_augmentations():
            self._apply_data_augmentations()
        else:
            print("Skipping data augmentations (not configured)")
        
        if self._should_apply_transformations():
            self._apply_data_transformations()
        else:
            print("Skipping data transformations (not configured)")
    
    def _finalization_stage(self):
        print("\n=== Stage 4: Finalization ===")
        
        # Make sure we have the files to save
        if not self.train_files and not self.val_files:
            print("  No processed files to save!")
            return
        
        print(f"  Saving {len(self.train_files)} train files and {len(self.val_files)} val files")
        
        self.output_coordinator.save_processed_data(
            self.train_files, 
            self.val_files, 
            self.config["dataset_paths"]["processed_data"]
        )
        
        # Verify final files exist
        train_final = os.path.join(self.config["dataset_paths"]["processed_data"], "train", "train_data.csv")
        val_final = os.path.join(self.config["dataset_paths"]["processed_data"], "val", "val_data.csv")
        
        if os.path.exists(train_final):
            train_count = self.file_processor.count_samples_in_file(train_final)
            print(f"  ✓ Final train file created: {train_count} samples")
        else:
            print(f"  ✗ Final train file missing: {train_final}")
        
        if os.path.exists(val_final):
            val_count = self.file_processor.count_samples_in_file(val_final)
            print(f"  ✓ Final val file created: {val_count} samples")
        else:
            print(f"  ✗ Final val file missing: {val_final}")
    
    def _should_apply_augmentations(self):
        augmentation_config = self.config.get("augmentations", {})
        if not augmentation_config:
            return False
            
        x_sorting = augmentation_config.get("x_sorting", {})
        y_sorting = augmentation_config.get("y_sorting", {})
        point_wise_permutation = augmentation_config.get("point_wise_permutation_augmentation_pct", 0)
        tetrahedron_wise_permutation = augmentation_config.get("tetrahedron_wise_permutation_augmentation_pct", 0)
        
        has_x_sorting = any(x_sorting.get(key) for key in x_sorting if x_sorting.get(key))
        has_y_sorting = any(y_sorting.get(key) for key in y_sorting if y_sorting.get(key))
        has_permutations = point_wise_permutation > 0 or tetrahedron_wise_permutation > 0

        return has_x_sorting or has_y_sorting or has_permutations
    
    def _should_apply_transformations(self):
        transformation_config = self.config.get("transformations")
        
        if transformation_config is None:
            return False
        
        if isinstance(transformation_config, str):
            return bool(transformation_config.strip())
        elif isinstance(transformation_config, list):
            return len(transformation_config) > 0 and any(t for t in transformation_config if t)
        
        return False

    def _apply_data_augmentations(self):
        print("Applying data augmentations with streaming...")
        
        augmentation_config = self.config.get("augmentations", {})
        if not augmentation_config:
            print("  No augmentation config found")
            return
            
        x_sorting = augmentation_config.get("x_sorting", {})
        y_sorting = augmentation_config.get("y_sorting", {})
        point_wise_pct = augmentation_config.get("point_wise_permutation_augmentation_pct", 0)
        tetrahedron_wise_pct = augmentation_config.get("tetrahedron_wise_permutation_augmentation_pct", 0)
        
        active_augmentations = []
        
        # Check for sorting augmentations
        if x_sorting:
            for key, value in x_sorting.items():
                if value:
                    active_augmentations.append(f"x_sorting.{key}: {value}")
        if y_sorting:
            for key, value in y_sorting.items():
                if value:
                    active_augmentations.append(f"y_sorting.{key}: {value}")
        
        # Check for geometric augmentations
        if point_wise_pct > 0:
            active_augmentations.append(f"point_wise_permutation: {point_wise_pct}%")
        if tetrahedron_wise_pct > 0:
            active_augmentations.append(f"tetrahedron_wise_permutation: {tetrahedron_wise_pct}%")
        
        if active_augmentations:
            print(f"  Active augmentations: {', '.join(active_augmentations)}")
            self._apply_augmentations_to_files(self.train_files, 'train')
            self._apply_augmentations_to_files(self.val_files, 'val')
        else:
            print("  No augmentations configured")

    def _apply_data_transformations(self):
        transformation_config = self.config.get("transformations", [])
        
        if isinstance(transformation_config, str):
            transformation_list = [transformation_config] if transformation_config.strip() else []
        elif isinstance(transformation_config, list):
            transformation_list = [t for t in transformation_config if t]
        else:
            transformation_list = []
        
        if transformation_list:
            print(f"Applying data transformations: {', '.join(transformation_list)}")
            self._apply_transformations_to_files(self.train_files, 'train')
            self._apply_transformations_to_files(self.val_files, 'val')
        else:
            print("No transformations configured")
    
    def _apply_augmentations_to_files(self, file_list, split_name):
        if not file_list:
            return
        
        augmented_files = []
        
        for file_idx, file_path in enumerate(file_list):
            augmented_file = os.path.join(self.temp_dir, f"{split_name}_aug_{file_idx}.csv")
            first_write = True
            
            for chunk_df in pd.read_csv(file_path, chunksize=10_000):
                try:
                    augmented_chunk = AugmentationEngine.augment_data(chunk_df, self.config)
                    augmented_chunk.to_csv(augmented_file, mode='a', header=first_write, index=False)
                    first_write = False
                except Exception as e:
                    print(f"Warning: Augmentation failed for chunk: {e}")
                    chunk_df.to_csv(augmented_file, mode='a', header=first_write, index=False)
                    first_write = False
                
                del chunk_df
                gc.collect()
            
            augmented_files.append(augmented_file)
        
        if split_name == 'train':
            self.train_files = augmented_files
        else:
            self.val_files = augmented_files
    
    def _apply_transformations_to_files(self, file_list, split_name):
        if not file_list:
            return
        
        transformed_files = []
        
        for file_idx, file_path in enumerate(file_list):
            transformed_file = os.path.join(self.temp_dir, f"{split_name}_trans_{file_idx}.csv")
            first_write = True
            
            for chunk_df in pd.read_csv(file_path, chunksize=10_000):
                try:
                    transformed_chunk = TransformationEngine.transform_data(chunk_df, self.config)
                    transformed_chunk.to_csv(transformed_file, mode='a', header=first_write, index=False)
                    first_write = False
                except Exception as e:
                    print(f"Warning: Transformation failed for chunk: {e}")
                    chunk_df.to_csv(transformed_file, mode='a', header=first_write, index=False)
                    first_write = False
                
                del chunk_df
                gc.collect()
            
            transformed_files.append(transformed_file)
        
        if split_name == 'train':
            self.train_files = transformed_files
        else:
            self.val_files = transformed_files
    
    # Static methods for backward compatibility
    @staticmethod
    def augment_data(data: pd.DataFrame, config) -> pd.DataFrame:
        return AugmentationEngine.augment_data(data, config)
    
    @staticmethod
    def transform_data(data: pd.DataFrame, config) -> pd.DataFrame:
        return TransformationEngine.transform_data(data, config)