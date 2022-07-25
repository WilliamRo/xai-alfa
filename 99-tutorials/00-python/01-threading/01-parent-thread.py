import threading
import time



main_thread = threading.currentThread()

def show_parent_status():
  count_down = 5
  while count_down > 0:
    if not main_thread.is_alive(): count_down -= 1
    print(f'parent.is_alive = {main_thread.is_alive()}')
    time.sleep(1)

child = threading.Thread(target=show_parent_status)
child.start()

time.sleep(2)
print('>> main thread terminated.')








