import os
import pandas as pd
import gc
import subprocess  # Add this line
import warnings
warnings.filterwarnings('ignore')


class FileProcessor:
    """Utility class for file operations and data streaming."""
    
    @staticmethod
    def count_samples_in_files(file_list):
        """Count total samples across multiple files."""
        total = 0
        for file_path in file_list:
            if os.path.exists(file_path):
                try:
                    count = FileProcessor.count_samples_in_file(file_path)
                    total += count
                    print(f"    {os.path.basename(file_path)}: {count} samples")
                except Exception as e:
                    print(f"    Error counting {file_path}: {e}")
        return total

    @staticmethod
    def _validate_and_clean_chunk(chunk_df, expected_columns):
        """Validate and clean a chunk of data with float64 precision."""
        if chunk_df.empty:
            return chunk_df
        
        try:
            # Fix: expected_columns can be either int or list
            if isinstance(expected_columns, list):
                expected_count = len(expected_columns)
                expected_names = expected_columns
            else:
                expected_count = expected_columns
                expected_names = None
            
            actual_count = len(chunk_df.columns)
            
            if actual_count != expected_count:
                print(f"        Column mismatch: expected {expected_count}, got {actual_count}")
                # If we have extra columns, keep only the first expected_count
                if actual_count > expected_count:
                    chunk_df = chunk_df.iloc[:, :expected_count]
                    print(f"        Trimmed to first {expected_count} columns")
                else:
                    print(f"        WARNING: Missing columns, keeping {actual_count} columns")
            
            # Basic data cleaning
            chunk_df = chunk_df.dropna(how='all')
            
            # Convert data types properly - using float64 for maximum precision
            coordinate_columns = chunk_df.columns[:-2]  # All except last 2 columns
            
            # Convert coordinate columns to float64
            for col in coordinate_columns:
                chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce').astype('float64')  # Changed to float64
            
            # Convert volume column to float64
            if len(chunk_df.columns) >= 2:
                volume_col = chunk_df.columns[-2]  # Second to last column
                chunk_df[volume_col] = pd.to_numeric(chunk_df[volume_col], errors='coerce').astype('float64')  # Changed to float64
            
            # Convert label column to int32
            if len(chunk_df.columns) >= 1:
                label_col = chunk_df.columns[-1]  # Last column
                chunk_df[label_col] = pd.to_numeric(chunk_df[label_col], errors='coerce').astype('int32')
            
            # Fill any remaining NaN values
            chunk_df = chunk_df.fillna(0)
            
            # Remove any rows that became completely invalid
            chunk_df = chunk_df.dropna(how='all')
            
            return chunk_df
            
        except Exception as e:
            print(f"        Error validating chunk: {e}")
            return pd.DataFrame()

    @staticmethod
    def combine_files_streaming(source_files, output_file, chunk_size=50_000):
        """Combine multiple files with streaming and maximum float64 precision."""
        expected_samples = FileProcessor.count_samples_in_files(source_files)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Remove existing output file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"  Removed existing output file: {output_file}")
        
        print(f"  Combining {len(source_files)} files into {os.path.basename(output_file)}")
        
        total_written = 0
        first_file = True
        expected_column_count = None
        
        for file_idx, source_file in enumerate(source_files):
            try:
                if not os.path.exists(source_file):
                    print(f"    Skipping missing file: {source_file}")
                    continue
                    
                print(f"    Processing file {file_idx + 1}/{len(source_files)}: {os.path.basename(source_file)}")
                
                file_written = 0
                for chunk_df in pd.read_csv(source_file, chunksize=chunk_size):
                    if first_file and expected_column_count is None:
                        expected_column_count = len(chunk_df.columns)
                        print(f"      Expected columns: {expected_column_count}")
                    
                    # Use count instead of list for validation
                    chunk_df = FileProcessor._validate_and_clean_chunk(chunk_df, expected_column_count)
                    
                    if len(chunk_df) > 0:
                        # Write with maximum precision for float64
                        chunk_df.to_csv(output_file, mode='a', header=first_file, index=False,
                                    float_format='%.17g')  # Maximum precision for float64
                        first_file = False
                        file_written += len(chunk_df)
                        total_written += len(chunk_df)
                    
                    del chunk_df
                    gc.collect()
                
                print(f"      Added {file_written} rows from {os.path.basename(source_file)}")
                
            except Exception as e:
                print(f"    Error processing {source_file}: {e}")
                continue
        
        print(f"  Final combined file: {total_written} total rows")
        
        # Validate the final file
        if total_written > 0 and os.path.exists(output_file):
            FileProcessor.validate_and_fix_file_consistency(output_file)
            final_count = FileProcessor.count_samples_in_file(output_file)
            print(f"  Verification: {final_count} rows in final file")
            
            if abs(final_count - expected_samples) > 100:
                print(f"  WARNING: Expected ~{expected_samples} but got {final_count}")
        else:
            print(f"  ERROR: No data was written to {output_file}")
            return False
        
        return True

    @staticmethod
    def validate_and_fix_file_consistency(file_path, chunk_size=10_000):
        """Validate and fix file consistency."""
        if not os.path.exists(file_path):
            print(f"  File does not exist: {file_path}")
            return False
        
        try:
            print(f"Validating data consistency in {os.path.basename(file_path)}")
            
            # Define expected columns
            expected_columns = [
                'T1_v1_x', 'T1_v1_y', 'T1_v1_z', 'T1_v2_x', 'T1_v2_y', 'T1_v2_z',
                'T1_v3_x', 'T1_v3_y', 'T1_v3_z', 'T1_v4_x', 'T1_v4_y', 'T1_v4_z',
                'T2_v1_x', 'T2_v1_y', 'T2_v1_z', 'T2_v2_x', 'T2_v2_y', 'T2_v2_z',
                'T2_v3_x', 'T2_v3_y', 'T2_v3_z', 'T2_v4_x', 'T2_v4_y', 'T2_v4_z',
                'IntersectionVolume', 'HasIntersection'
            ]
            
            print(f"  Expected columns: {len(expected_columns)}")
            
            temp_output = file_path + "_temp"
            total_processed = 0
            first_chunk = True
            
            for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
                # Use the count instead of list for validation
                validated_chunk = FileProcessor._validate_and_clean_chunk(chunk_df, len(expected_columns))
                
                if len(validated_chunk) > 0:
                    # Set correct column names if they don't match
                    if list(validated_chunk.columns) != expected_columns:
                        if len(validated_chunk.columns) == len(expected_columns):
                            validated_chunk.columns = expected_columns
                    
                    validated_chunk.to_csv(temp_output, mode='a', header=first_chunk, 
                                        index=False, float_format='%.17g')
                    first_chunk = False
                    total_processed += len(validated_chunk)
                
                del chunk_df, validated_chunk
                gc.collect()
            
            if total_processed > 0:
                # Replace original with validated file
                os.replace(temp_output, file_path)
                print(f"  Validated and fixed {total_processed} rows")
                return True
            else:
                print(f"  No output file created during validation")
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                return False
                
        except Exception as e:
            print(f"  Error during validation: {e}")
            return False

    @staticmethod
    def count_samples_in_file(file_path):
        """Count samples in a single file."""
        if not os.path.exists(file_path):
            return 0
        
        try:
            # Try to use wc -l for fast counting on Unix systems
            if os.name == 'posix':
                try:
                    result = subprocess.run(['wc', '-l', file_path], 
                                        capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        line_count = int(result.stdout.split()[0])
                        return max(0, line_count - 1)  # Subtract 1 for header
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    pass  # Fall back to pandas method
            
            # Fallback to pandas counting
            total_rows = 0
            chunk_size = 10000
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                total_rows += len(chunk)
            return total_rows
            
        except Exception as e:
            print(f"Error counting samples in {file_path}: {e}")
            return 0
