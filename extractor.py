from argparse import ArgumentParser, Namespace
from pydub import AudioSegment
import torch
import torchaudio
import numpy as np
import math

from .model.net import FCN, FCN05, ShortChunkCNN_Res

def get_audio(audio_path):
    if audio_path[-3:] in ('m4a', 'aac'):
        song = AudioSegment.from_file(audio_path, 'm4a').set_frame_rate(16000).set_channels(1)._data
    else:
        song = AudioSegment.from_file(audio_path, audio_path[-3:]).set_frame_rate(16000).set_channels(1)._data
    decoded = np.frombuffer(song, dtype=np.int16) / 32768
    audio_tensor = torch.Tensor(decoded).unsqueeze(0)
    # else:
        # waveform, sr = torchaudio.load(mp3_path)
        # downsample_resample = torchaudio.transforms.Resample(sr, 16000)
        # audio_tensor = downsample_resample(waveform)
        # if audio_tensor.shape[0] == 2:
        #     audio_tensor = (audio_tensor[0:1,:] + audio_tensor[1:2,:]) /2
    return audio_tensor, audio_tensor.shape[1]

def load_model(models):
    if models == "FCN05":
        input_length= 8000
        model = FCN05()
        checkpoint_path = (
            f"weights/FCN05-roc_auc=0.8552-pr_auc=0.3344.ckpt"
        )
    elif models == "FCN037":
        input_length= 59049
        model = ShortChunkCNN_Res()
        checkpoint_path = (
            f"weights/ShortChunkCNN037-roc_auc=0.8948-pr_auc=0.4039.ckpt"
        )
    elif models == "FCN29":
        input_length= 464000
        model = FCN()
        checkpoint_path = (
            f"weights/FCN29-roc_auc=0.9025-pr_auc=0.4342.ckpt"
        )
    return input_length, model, checkpoint_path

def make_frames_of_batch(audio_tensor, input_length, sr=16000, target_fps=2):
    num_frame = int(audio_tensor.shape[1] / input_length)
    hop_size = int(sr / target_fps)
    audio_length = audio_tensor.shape[-1]
    num_slice = (audio_length-input_length) // hop_size
    hop_positions = [int(x*sr/target_fps) for x in range(num_slice+1)]
    split = [audio_tensor[:,i:i+input_length] for i in hop_positions]
    batch_audio = torch.stack(split, dim=1)
    return batch_audio

def make_frames(audio_tensor, input_length):
    num_frame = int(audio_tensor.shape[1] / input_length)
    audio_tensor = torch.reshape(audio_tensor[0,:num_frame*input_length],(num_frame,input_length))
    return audio_tensor

def make_audio_batch(audio_tensor, input_length, sampleing_rate = 16000, target_fps=3):
    # num_frame = int(audio_length / input_length)
    padded_audio = torch.nn.functional.pad(audio_tensor, (input_length//2, input_length//2), mode='constant')
    audio_length = padded_audio.shape[-1]
    num_slice = int((audio_length-input_length) // (sampleing_rate / target_fps))
    hop_positions = [int(x*sampleing_rate/target_fps) for x in range(num_slice+1)]
    split = [padded_audio[i:i+input_length] for i in hop_positions]
    batch_audio = torch.stack(split)
    return batch_audio

def get_frame_embeddings(mp3_path, model_type, target_fps, device='cuda'):
    results = []
    audio, _ = get_audio(mp3_path)
    input_length, model, checkpoint_path = load_model(model_type)
    batch_audio = make_audio_batch(audio[0], input_length, target_fps=target_fps)
    batch_audio = torch.split(batch_audio, 100)
    
    state_dict = torch.load("music_embedder/"+ checkpoint_path, map_location=torch.device('cpu'))
    new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
    new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    for i in batch_audio:
        with torch.no_grad():
            # _, embeddings = model(i)
            _, embeddings = model(i.to(device))
            results.append(embeddings)
    results = torch.cat(results)
    return results

# def embedding_extractor(audio_path, model_types, target_fps=3):
#     audio, audio_length = get_audio(audio_path)
#     # labels = np.load('dataset/mtat/split/tags.npy')

#     _, model, checkpoint_path = load_model(model_types)

#     # audio_input = make_frames(audio, input_length)
#     state_dict = torch.load("Music_DeepEmbedding_Extractor/"+checkpoint_path, map_location=torch.device('cpu'))

#     new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
#     new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
#     model.load_state_dict(new_state_dict)
#     for i in range(1,8):
#         for j in (1,3):
#             getattr(getattr(model, 'layer{}'.format(i)), 'conv_{}'.format(j)).padding = (1,0)
    
#     view_size = 130304
#     audio_batch_size = 100
#     dummy = torch.zeros((1, audio.shape[1] + view_size *2))
#     dummy[:,view_size//2:view_size//2+audio.shape[1]] = audio
#     audio = dummy
#     num_frame = math.ceil(audio_length/ 16000 * target_fps)
#     num_batch = math.ceil(num_frame / audio_batch_size)

#     model = model.to('cuda')
#     model.eval() 
#     total_embeddings = []
#     with torch.no_grad():
        
#     # _, embeddings = model(audio_input)
#         for batch_i in range(num_batch):
#             start_idx = int(batch_i * audio_batch_size * 16000/target_fps)
#             if batch_i == num_batch -1:
#                 num_segments = num_frame % audio_batch_size
#             else:
#                 num_segments = audio_batch_size
#             batch_audio = torch.stack([audio[0,start_idx+int(i*16000/target_fps):start_idx+int(i*16000/target_fps)+view_size ] for i in range(num_segments) ])
#             embeddings = model.get_emb(batch_audio.to('cuda'))
#             total_embeddings.append(embeddings[:,0,:])
#     return torch.cat(total_embeddings)

def main(args) -> None:
    embedding = embedding_extractor(args.audio_path, args.models)
    print(embedding)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--models", default="FCN05", type=str, choices=["FCN05", "FCN037", "FCN29"])
    parser.add_argument("--audio_path", default="dataset/mtat/test_mp3/sample2.mp3", type=str)
    args = parser.parse_args()
    main(args)
