'''
huggingface-cli download svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP --repo-type dataset --revision main --include "genshin_impact_CHONGYUN_images_and_texts/*" --local-dir .

python run_cli_app.py
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
