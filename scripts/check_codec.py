"""Manual Windows codec diagnostic; safe to import on other platforms."""
import os


def main():
    if os.name != "nt":
        raise SystemExit("This diagnostic is for Windows only")
    dll_dir = os.getenv("FFMPEG_DLL_PATH", r"C:\ffmpeg7\bin")
    with os.add_dll_directory(dll_dir):
        import torch
        import torchcodec
        print("torch:", torch.__version__)
        print("cuda:", torch.cuda.is_available())
        print("torchcodec:", torchcodec.__version__)


if __name__ == "__main__":
    main()
