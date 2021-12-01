import io
from pathlib import Path
import select
import subprocess as sp
from typing import Dict, Tuple, Optional, IO
import nussl
import os
import sys

def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        # `select` syscall will wait until one of the file descriptors has content.
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()

class DemucsModel:
    
    def __init__(self,mix,model,in_path,out_path):
        self.mix = mix
        self.model = model
        self.in_path = in_path
        self.out_path = out_path
        
    def __call__(self):
        self.mix.write_audio_to_file(os.path.join(self.in_path,"mix.wav"))
        self.separate()
        preds = {}
        for stem in ["bass","drums","other","vocals"]:
            preds[stem] = nussl.AudioSignal()
            preds[stem].load_audio_from_file(os.path.join(self.out_path,self.model,"mix",f"{stem}.wav"))
        return [preds["bass"]+preds["drums"]+preds["other"], preds["vocals"]]
    
    
    def find_files(self, extensions=["wav"]):
        out = []
        for file in Path(self.in_path).iterdir():
            if file.suffix.lower().lstrip(".") in extensions:
                out.append(file)
        return out
    
    def separate(self):
        cmd = ["python3", "-m", "demucs.separate", "-o", str(self.out_path), "-n", self.model]
        files = [str(f) for f in self.find_files()]
        if not files:
            print(f"No valid audio files in {self.in_path}")
            return
        print("Going to separate the files:")
        print('\n'.join(files))
        print("With command: ", " ".join(cmd))
        p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
        copy_process_streams(p)
        p.wait()
        if p.returncode != 0:
            print("Command failed, something went wrong.")