import subprocess as sp
import os

sp.Popen(["mkdir","frames_rede"])
sp.run(["python3","kuramoto_rede.py"])
os.chdir("frames_rede")
sp.run(["ffmpeg","-framerate","5","-i","%03d.png","simulation.mp4"])

k = 0
name = '00{}.png'.format(k)
while os.path.exists(name):
    os.remove(name)

    k = k+1
    if k < 10:
        name = '00{}.png'.format(k)
    elif 9 < k < 100:
        name = '0{}.png'.format(k)
    else:
        name = '{}.png'.format(k)

os.chdir("../")
sp.Popen(["mv","frames_rede/simulation.mp4","simulation.mp4"])

print('Simulacao feita.')

sp.Popen(["rmdir", "frames_rede"])
