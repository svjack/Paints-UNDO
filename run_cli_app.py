'''
huggingface-cli download svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP --repo-type dataset --revision main --include "genshin_impact_CHONGYUN_images_and_texts/*" --local-dir .

python run_cli_app.py

import os
import shutil
from moviepy.editor import VideoFileClip
from pathlib import Path

def process_videos_and_copy_files(source_dir, target_dir):
    """
    处理源目录中的所有MP4文件并拷贝其他文件到目标目录
    
    参数:
        source_dir: 源目录路径
        target_dir: 目标目录路径
    """
    # 确保目标目录存在
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # 遍历源目录中的所有文件
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            source_path = os.path.join(root, filename)
            relative_path = os.path.relpath(source_path, source_dir)
            target_path = os.path.join(target_dir, relative_path)
            
            # 确保目标文件的目录存在
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # 处理MP4文件
            if filename.lower().endswith('.mp4'):
                try:
                    print(f"正在处理视频文件: {source_path}")
                    
                    # 使用MoviePy处理视频
                    clip = VideoFileClip(source_path)
                    clip.write_videofile(target_path)
                    clip.close()
                    
                    print(f"视频已保存到: {target_path}")
                except Exception as e:
                    print(f"处理视频 {source_path} 时出错: {str(e)}")
            else:
                # 拷贝非MP4文件
                try:
                    shutil.copy2(source_path, target_path)
                    print(f"已拷贝文件: {source_path} -> {target_path}")
                except Exception as e:
                    print(f"拷贝文件 {source_path} 时出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
    source_directory = "genshin_impact_CHONGYUN_Paints_UNDO_UnChunked"
    target_directory = "Genshin_Impact_CHONGYUN_Paints_UNDO_UnChunked"
    
    process_videos_and_copy_files(source_directory, target_directory)
    print("所有文件处理完成!")

'''

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# 配置路径
source_dir = Path("genshin_impact_CHONGYUN_images_and_texts")
output_dir = Path("genshin_impact_CHONGYUN_Paints_UNDO")
output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

# 固定视频提示
video_prompt = "masterpiece, best quality, highly detailed"

# 获取所有.png文件
png_files = list(source_dir.glob("*.png"))

# 处理每个.png文件
for png_file in tqdm(png_files, desc="Processing files"):
    # 获取文件名（不带扩展名）
    stem = png_file.stem
    
    # 对应的.txt文件路径
    txt_file = png_file.with_suffix(".txt")
    
    # 输出.mp4文件路径
    mp4_file = png_file.with_suffix(".mp4")
    
    # 构建命令行命令
    cmd = f"python cli_app.py {png_file} --video_prompt \"{video_prompt}\" --output \"{mp4_file}\""
    
    # 执行命令（这里只是打印，实际使用时取消注释os.system）
    print(f"\nExecuting: {cmd}")
    os.system(cmd)
    
    # 检查.txt文件是否存在
    if txt_file.exists():
        # 拷贝.txt文件到输出目录
        shutil.copy2(txt_file, output_dir / txt_file.name)
        print(f"Copied: {txt_file} -> {output_dir / txt_file.name}")
        # 删除原始.txt文件
        #txt_file.unlink()
        print(f"Deleted original: {txt_file}")
    else:
        print(f"Warning: No corresponding .txt file found for {png_file}")
    
    # 检查.mp4文件是否生成（在实际使用中需要等待命令执行完成）
    if mp4_file.exists():
        # 拷贝.mp4文件到输出目录
        shutil.copy2(mp4_file, output_dir / mp4_file.name)
        print(f"Copied: {mp4_file} -> {output_dir / mp4_file.name}")
        # 删除原始.mp4文件
        mp4_file.unlink()
        print(f"Deleted original: {mp4_file}")
    else:
        print(f"Warning: MP4 file not generated for {png_file}")
    
    # 删除原始.png文件（可选，如果你确定不再需要）
    # png_file.unlink()
    # print(f"Deleted original: {png_file}")

print("\nProcessing complete!")
