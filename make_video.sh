ffmpeg -i imgs/process/%d.png -vf fps=20 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx264 -y -an video.mp4