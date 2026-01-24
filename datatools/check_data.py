import os
import sys
from collections import defaultdict
import shutil
import tempfile

def read_file_with_encoding(file_path):
    """尝试用多种编码读取文件"""
    encodings = ['utf-8', 'gbk', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = [line.rstrip('\n') for line in file]
            return lines, encoding, None
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return None, None, str(e)
    
    return None, None, "无法使用任何支持的编码读取文件"

def find_duplicate_lines_in_file(file_path):
    """
    检查单个文件中重复的行
    """
    duplicates = defaultdict(list)  # 存储行内容和对应的行号
    duplicate_count = 0
    unique_duplicates = set()  # 用于统计不同的重复内容
    
    lines, encoding, error = read_file_with_encoding(file_path)
    
    if error:
        print(f"  错误: {error}")
        return 0, duplicates, unique_duplicates, None, None
    
    for line_num, line in enumerate(lines, start=1):
        if line in duplicates:
            if len(duplicates[line]) == 1:
                # 第一次发现重复
                unique_duplicates.add(line)
        duplicates[line].append(line_num)
    
    # 统计重复数量
    duplicate_count = len(unique_duplicates)
    
    return duplicate_count, duplicates, unique_duplicates, lines, encoding

def remove_duplicate_lines(file_path, lines, encoding, backup=True):
    """
    删除文件中的重复行（只保留第一次出现的行）
    """
    # 如果备份选项开启，创建备份文件
    if backup:
        backup_path = file_path + '.bak'
        shutil.copy2(file_path, backup_path)
        print(f"  已创建备份文件: {backup_path}")
    
    # 使用集合跟踪已出现的行，只保留第一次出现的行
    seen = set()
    unique_lines = []
    
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    
    # 写入去重后的内容
    try:
        with open(file_path, 'w', encoding=encoding) as file:
            for line in unique_lines:
                file.write(line + '\n')
        
        removed_count = len(lines) - len(unique_lines)
        return True, removed_count, None
    except Exception as e:
        return False, 0, str(e)

def process_file(file_path, remove_duplicates=False, backup=True):
    """
    处理单个文件
    """
    print(f"\n处理文件: {os.path.basename(file_path)}")
    
    duplicate_count, duplicates, unique_duplicates, lines, encoding = find_duplicate_lines_in_file(file_path)
    
    if lines is None:
        print(f"  错误: 无法读取文件")
        return 0, 0
    
    if duplicate_count == 0:
        print(f"  ✓ 文件中没有重复的行")
        return 0, 0
    else:
        print(f"  ! 找到 {duplicate_count} 处不同的重复内容")
        
        for line_content in unique_duplicates:
            line_numbers = duplicates[line_content]
            if len(line_numbers) > 1:
                # print(f"    重复内容: '{line_content}'")
                print(f"    重复行号: {line_numbers}")
                print(f"    重复次数: {len(line_numbers)} 次")
                print(f"    {'-' * 40}")
        
        if remove_duplicates:
            print(f"\n  → 正在删除重复行...")
            success, removed_count, error = remove_duplicate_lines(file_path, lines, encoding, backup)
            
            if success:
                print(f"  ✓ 已删除 {removed_count} 行重复内容")
                print(f"    原文件有 {len(lines)} 行，去重后有 {len(lines) - removed_count} 行")
            else:
                print(f"  ✗ 删除失败: {error}")
        
        return duplicate_count, removed_count if remove_duplicates else 0

def process_directory(directory_path, remove_duplicates=False, backup=True):
    """
    处理目录中的所有txt文件
    """
    if not os.path.exists(directory_path):
        print(f"错误: 目录 '{directory_path}' 不存在")
        return
    
    if not os.path.isdir(directory_path):
        print(f"错误: '{directory_path}' 不是目录")
        return
    
    txt_files = [f for f in os.listdir(directory_path) 
                if f.lower().endswith('.txt')]
    
    if not txt_files:
        print(f"在目录 '{directory_path}' 中未找到txt文件")
        return
    
    print(f"在目录 '{directory_path}' 中找到 {len(txt_files)} 个txt文件")
    print("=" * 60)
    
    total_duplicates_all_files = 0
    total_removed_all_files = 0
    
    for txt_file in txt_files:
        file_path = os.path.join(directory_path, txt_file)
        duplicates_count, removed_count = process_file(file_path, remove_duplicates, backup)
        total_duplicates_all_files += duplicates_count
        total_removed_all_files += removed_count
    
    print("\n" + "=" * 60)
    if remove_duplicates:
        print(f"统计完成! 所有文件中:")
        print(f"  - 总共找到 {total_duplicates_all_files} 处不同的重复内容")
        print(f"  - 总共删除了 {total_removed_all_files} 行重复内容")
    else:
        print(f"统计完成! 所有文件中总共找到 {total_duplicates_all_files} 处不同的重复内容")

def main():
    """
    主函数
    """
    print("文本文件重复行处理工具")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        # 交互式模式
        directory_path = input("请输入要检查的目录路径（直接回车使用当前目录）: ").strip()
        if directory_path == "":
            directory_path = "."
        
        if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
            print(f"错误: 目录 '{directory_path}' 不存在或不是有效的目录")
            return
        
        action = input("请选择操作: [1]只检查重复行 [2]检查并删除重复行 (默认1): ").strip()
        remove_duplicates = (action == "2")
        
        backup_option = "y"
        if remove_duplicates:
            backup_option = input("删除前是否创建备份文件? [y/n] (默认y): ").strip().lower()
            if backup_option == "":
                backup_option = "y"
        
        backup = (backup_option == "y")
        
        process_directory(directory_path, remove_duplicates, backup)
    else:
        # 命令行参数模式
        import argparse
        
        parser = argparse.ArgumentParser(description='检查并删除txt文件中的重复行')
        parser.add_argument('directory', help='要处理的目录路径')
        parser.add_argument('-r', '--remove', action='store_true', 
                          help='删除重复行（只保留第一次出现的行）')
        parser.add_argument('-n', '--no-backup', action='store_true',
                          help='删除重复行时不创建备份文件')
        
        args = parser.parse_args()
        
        directory_path = args.directory
        remove_duplicates = args.remove
        backup = not args.no_backup
        
        if not (os.path.exists(directory_path) and os.path.isdir(directory_path)):
            print(f"错误: 目录 '{directory_path}' 不存在或不是有效的目录")
            return
        
        process_directory(directory_path, remove_duplicates, backup)

if __name__ == "__main__":
    main()