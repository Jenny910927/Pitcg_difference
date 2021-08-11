import argparse
from dataset_former_note import PitchDiffDataset
import os
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path')

    args = parser.parse_args()

    # These two lists contain the path to all training data pair
    json_paths = ["redire.json", "Sugar_Song.json", "Alone.json", "Christmas_Song.json", "Lemon.json"
        , "Liekki.json", "Marigold.json", "Trail.json", "Yesterday.json", "Pretender.json", "This_Earth.json"
        , "366_days.json", "Interstellar.json", "Moon.json", "Universe.json", "Wonder.json", "Warui.json"
        , "Yurayura.json", "Yurika.json"]

    # json_paths = ["Human.json",]


    new_json_paths = [os.path.join("json檔", json_path) for json_path in json_paths]

    audio_paths = ["redire.mp4", "Sugar_Song.mp3", "Alone.mp3", "Christmas_Song.mp3", "Lemon.mp3"
        , "Liekki.mp3", "Marigold.mp3", "Trail.mp3", "Yesterday.mp3", "Pretender.mp3", "This_Earth.mp3"
        , "366_days.mp3", "Interstellar.mp3", "Moon.mp3", "Universe.mp3", "Wonder.mp3", "Warui.mp3"
        , "Yurayura.mp3", "Yurika.mp3"]

    # audio_paths = ["Human.mp3",]

    new_audio_paths = [os.path.join("Youtube音檔", audio_path) for audio_path in audio_paths]
    

    dataset = PitchDiffDataset(json_paths=new_json_paths, audio_paths=new_audio_paths)

    with open(args.output_path, 'wb') as f:
        pickle.dump(dataset, f)
