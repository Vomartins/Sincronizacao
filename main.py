import subprocess as sp
import os

pergunta = str(input("Há uma força externa? (s/n) \n"))
if pergunta=='s' or pergunta=='S':
    print('##################################################\n')
    sp.Popen(["mkdir","frames_forc"])
    sp.run(["python3","kuramoto_forc.py"])
    os.chdir("frames_forc")
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
    sp.Popen(["mv","frames_forc/simulation.mp4","simulation.mp4"])

    print('Simulacao feita.')

    sp.Popen(["rmdir", "frames_forc"])

elif pergunta=='n' or pergunta=='N':
    pergunta1 = str(input("Diferentes constantes de acoplamento para cada dimensão? (s/n) \n"))
    if pergunta1=='s' or pergunta1=='S':
        print('##################################################\n')
        sp.Popen(["mkdir","frames_rede_1"])
        sp.run(["python3","kuramoto_rede_1.py"])
        os.chdir("frames_rede_1")
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
        sp.Popen(["mv","frames_rede_1/simulation.mp4","simulation.mp4"])

        print('Simulacao feita.')

        sp.Popen(["rmdir", "frames_rede_1"])

    elif pergunta1=='n' or pergunta1=='N':
        print('##################################################\n')
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
    else:
        print('##################################################\n')
        print('Opção inválida!')
        print('##################################################\n')    
        sp.run(["python3","main.py"])
else:
    print('##################################################\n')
    print('Opção inválida!')
    print('##################################################\n')    
    sp.run(["python3","main.py"])
