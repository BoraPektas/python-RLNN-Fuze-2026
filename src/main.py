#!/usr/bin/env python3
"""
main.py: Project entry point — Launches the GUI only.
(True PN Architecture version: Provides the AI with LOS rate, closing velocity, etc.)
"""
import sys
import os

if __name__ == "__main__":
    # Add src to python path if run from root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from gui import main_menu
        print("GUI Starting...")
        main_menu()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)