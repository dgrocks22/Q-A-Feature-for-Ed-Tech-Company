import os

def create_child_process():
    pid = os.fork()
    if pid > 0:
        print(f"Parent process{os.getpid()}  created.")
    if pid <= 0:
        print(f"Child process {os.getpid()} created.")
        os._exit(0)
for _ in range(3):
    create_child_process()
    