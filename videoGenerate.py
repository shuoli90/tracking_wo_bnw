# import cv2
# import numpy as np
# import skvideo.io

# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# title = 'kf'

# img1 = cv2.imread(f'images_{title}/seq1.png')

# height, width, layers =  img1.shape

# out_video =  np.empty([100, height, width, 3], dtype = np.uint8)
# out_video =  out_video.astype(np.uint8)

# for seq in range(1,101):
#     img = cv2.imread(f'images_{title}/seq{seq}.png')
#     out_video[seq-1] = img

# # Writes the the output image sequences in a video file
# skvideo.io.vwrite(f"images_{title}/video.mp4", out_video)

# import numpy as np

# import skvideo.datasets

# filename = skvideo.datasets.bigbuckbunny()

# vid_in = skvideo.io.FFmpegReader(filename)
# data = skvideo.io.ffprobe(filename)['video']
# rate = data['@r_frame_rate']
# T = np.int(data['@nb_frames'])

# vid_out = skvideo.io.FFmpegWriter("corrupted_video.mp4", inputdict={
#       '-r': rate,
#     },
#     outputdict={
#       '-vcodec': 'libx264',
#       '-pix_fmt': 'yuv420p',
#       '-r': rate,
# })
# for idx, frame in enumerate(vid_in.nextFrame()):
#   print("Writing frame %d/%d" % (idx, T))
#   if (idx >= (T/2)) & (idx <= (T/2 + 10)):
#     frame = np.random.normal(128, 128, size=frame.shape).astype(np.uint8)
#   vid_out.writeFrame(frame)
# vid_out.close()

import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--title', type=str, default='05',
                    help='an integer for the accumulator')

args = parser.parse_args()
title = args.title

img1 = cv2.imread(f'images_{title}/seq1.png')

height, width, layers =  img1.shape

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video=cv2.VideoWriter(f'images_{title}/video.mp4', fourcc, 15,(width,height))

for seq in range(1,101):
    img = cv2.imread(f'images_{title}/seq{seq}.png')
    video.write(img)

cv2.destroyAllWindows()
video.release()