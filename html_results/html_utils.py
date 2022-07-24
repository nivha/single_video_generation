import os
import jinja2


def parse_vid_section(vid_section):
    if len(vid_section) == 7:
        return vid_section
    if len(vid_section) == 6:
        return vid_section + (None,)
    if len(vid_section) == 5:
        return vid_section + (True, None)
    if len(vid_section) == 4:
        return vid_section + (30, True, None)


def fix_video_folders(vid_folders, base_dir):
    """
    input_structure:
    [
        (
            title,
            [frames_folder, frames_folder, ...],
            videos_color, [flex|block], max_column_width, full_width, max_frames
        ),
        ...
    ]

    output structure:
    (
       title,
       [
          (frames_folder, vid_global_index, vid_length, vid_color),
           ...,
       ],
       flex|block (initial display),
       max_column_width,
       full_width (whether each video takes all place (True), or show in its original resolution (False))
    )
    """
    global_vids = []
    vid_lengths = []

    vid_global_idx = 0
    for vid_section in vid_folders:
        title, folders, vids_color, section_initial_display, max_column_width, full_width, max_frames = parse_vid_section(vid_section)
        title_vids = []
        for i, vid_folder in enumerate(folders):
            vid_length = len(os.listdir(os.path.join(base_dir, vid_folder)))
            vid_length = vid_length if max_frames is None else min(vid_length, max_frames)


            title_vids.append(
                (
                    vid_folder,
                    vid_global_idx,
                    vid_length,
                    # vids_color if i > 0 else 'blue',
                    vids_color,
                )
            )
            vid_lengths.append(vid_length)
            vid_global_idx += 1

        global_vids.append(
            (
                title,
                title_vids,
                section_initial_display,
                max_column_width,
                full_width,
            )
        )

    return global_vids, vid_lengths, vid_global_idx + 1


def create_results_html(vid_folders, base_dir, html_filename, frame_rate=10, play_all_on_load=True):
    """
        full_width - whether the videos take all possible place, or show in original resolution
        column_max_width - affects how many videos get in one row (roughly, 40 --> 2 videos, 30 --> 3 videos in a row)
    """
    vids, video_lengths, N = fix_video_folders(vid_folders, base_dir)
    start_indexes = [0] * N
    loop_through = [0] * N

    # render html and save to file
    TEMPLATE_FILE = r"html_results/html_template.html"
    templateLoader = jinja2.FileSystemLoader(searchpath="./")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template(TEMPLATE_FILE)
    outputText = template.render(video_lengths=video_lengths,
                                 start_indexes=start_indexes,
                                 loop_through=loop_through,
                                 vids=vids,
                                 images_paths=[],
                                 frame_rate=frame_rate,
                                 play_all_on_load=play_all_on_load,
                                 )
    # save html to file
    output_path = os.path.join(base_dir, html_filename)
    with open(output_path, 'w') as f:
        f.write(outputText)
        print('saved to:', output_path)
