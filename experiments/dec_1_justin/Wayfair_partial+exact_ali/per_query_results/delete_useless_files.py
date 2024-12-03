import os
import glob

def delete_files():
    # Get all subdirectories in the current directory
    subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    total_dirs = len(subdirs)
    
    files_deleted = 0
    for idx, subdir in enumerate(subdirs, 1):
        # Files to delete
        target_files = [
            os.path.join(subdir, "eval_results.jsonl"),
            os.path.join(subdir, "trec_results_deduplicated.txt")
        ]
        
        # Try to delete each file
        for file_path in target_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                files_deleted += 1
                print(f"Deleted: {file_path}")
        
        # Print progress
        print(f"Progress: {idx}/{total_dirs} directories processed")
    
    print(f"\nComplete! Deleted {files_deleted} files in total.")

if __name__ == "__main__":
    delete_files()