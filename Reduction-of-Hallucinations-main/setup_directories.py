"""
Setup Directory Structure
==========================

Creates all necessary directories for the hallucination reduction pipeline.
Works on Windows, Mac, and Linux.

Usage:
    python setup_directories.py
"""

from pathlib import Path
import sys


def create_directory_structure():
    """Create all required directories for the project."""
    
    directories = [
        # Data directories
        "Sets",
        "phase1_data",
        "phase1_data/sft",
        "phase1_data_medhal",
        "phase1_data_medhal/sft",
        "phase2_data",
        "phase2_data/dpo",
        
        # Processing directories
        "processed",
        "processed_full",
        
        # Model directories
        "models",
        "models/sft_specialist",
        "models/dpo_hallucination_resistant",
        
        # Document directories
        "Documents",
        "Guides",
    ]
    
    print("=" * 70)
    print("Setting Up Directory Structure")
    print("=" * 70)
    print()
    
    created = []
    already_exists = []
    
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            already_exists.append(dir_path)
            print(f"  ✓ {dir_path} (already exists)")
        else:
            path.mkdir(parents=True, exist_ok=True)
            created.append(dir_path)
            print(f"  + {dir_path} (created)")
    
    print()
    print("=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print(f"Created: {len(created)} directories")
    print(f"Already existed: {len(already_exists)} directories")
    print()
    
    if created:
        print("Newly created directories:")
        for d in created:
            print(f"  - {d}")
        print()
    
    print("Directory structure is ready for:")
    print("  ✓ Phase 1 (SFT) data processing")
    print("  ✓ Phase 2 (DPO) data generation")
    print("  ✓ Model training and storage")
    print()


def verify_structure():
    """Verify that all critical directories exist."""
    critical_dirs = [
        "Sets",
        "phase1_data/sft",
        "phase2_data/dpo",
        "models",
    ]
    
    all_exist = True
    for dir_path in critical_dirs:
        if not Path(dir_path).exists():
            print(f"✗ Missing critical directory: {dir_path}", file=sys.stderr)
            all_exist = False
    
    return all_exist


def main():
    """Main entry point."""
    print()
    create_directory_structure()
    
    if verify_structure():
        print("✓ All critical directories verified!")
        print()
        print("Next steps:")
        print("  1. Place your raw data in the Sets/ directory")
        print("  2. Run Phase 1 preprocessing:")
        print("     python preprocess_data.py --generate-adversarial")
        print("  3. Generate Phase 2 data:")
        print("     python generate_phase2_data.py")
        print("  4. Train Stage A (SFT):")
        print("     python stage_a_sft_training.py")
        print("  5. Train Stage B (DPO):")
        print("     python stage_b_dpo_training.py")
        print()
        sys.exit(0)
    else:
        print("✗ Setup verification failed!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
