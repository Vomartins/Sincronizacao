import subprocess as sp
import os
import shutil

sp.run(["python.exe","kuramoto_rede.py"])
os.chdir("frames_rede")
sp.run("ffmpeg -framerate 5 -i %03d.png simulation.wmv")

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
shutil.move("frames_rede/simulation.wmv", "simulation.wmv")

print('Simulação feita.')