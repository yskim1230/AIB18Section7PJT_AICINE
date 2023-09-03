import os
import re


def delete_saved_model(model_name, saved_pth_dir_path):
    all_files = os.listdir(saved_pth_dir_path)
    deleted_files = []

    for file in all_files:
        if file.startswith(model_name):
            file_path = os.path.join(saved_pth_dir_path, file)
            os.remove(file_path)
            deleted_files.append(file)

    if deleted_files:
        print("다음 파일을 성공적으로 삭제했습니다.")
        for file in deleted_files:
            print(f"{file}")
    else:
        print(f"삭제할 모델({model_name}) 파일이 없습니다.")


def find_saved_model(model_name=None, load_pth_dir_path=""):
    def extract_epoch(filename):
        # "movies-" 뒤의 숫자를 추출하기 위한 정규표현식 사용
        match = re.search(r'movies-(\d+).pth', filename)
        if match:
            return int(match.group(1))
        return 0
    all_files = os.listdir(load_pth_dir_path)
    found_files = [file for file in all_files if file.startswith(model_name)]

    print(f"found_files{found_files}")

    if found_files:
        sorted_files = sorted(found_files, key=extract_epoch)
        print("다음 파일을 찾았습니다.")
        full_paths = []
        for idx, file in enumerate(sorted_files):
            # full_path = os.path.join(load_pth_dir_path, file)
            # full_paths.append(full_path)
            full_paths.append(file)
        return full_paths
    else:
        print("파일을 찾을 수 없습니다.")
        return None