def generate_percentage_report(profile_file="profile_results.prof"):
    """生成百分比分析报告"""
    
    # 加载分析结果
    stats = pstats.Stats(profile_file)
    
    # 获取所有统计信息
    stats.strip_dirs()
    
    # 计算总时间
    total_time = 0
    func_stats = {}
    
    # 解析统计信息
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        func_name = pstats.func_std_string(func)
        func_stats[func_name] = {
            'total_time': tt,
            'cumulative_time': ct,
            'call_count': nc,
            'primitive_call_count': cc,
            'callers': callers
        }
        total_time += ct
    
    # 生成报告
    report = {
        'generated_at': datetime.now().isoformat(),
        'total_time': total_time,
        'functions': []
    }
    
    # 按累积时间排序
    sorted_funcs = sorted(
        func_stats.items(),
        key=lambda x: x[1]['cumulative_time'],
        reverse=True
    )
    
    for func_name, stats in sorted_funcs[:100]:  # 只显示前20个
        func_report = {
            'function': func_name,
            'total_time': stats['total_time'],
            'cumulative_time': stats['cumulative_time'],
            'percentage': (stats['cumulative_time'] / total_time) * 100,
            'call_count': stats['call_count'],
            'average_time': stats['total_time'] / stats['call_count'] if stats['call_count'] > 0 else 0
        }
        report['functions'].append(func_report)
    
    # 打印报告
    print("\n" + "="*100)
    print("综合性能分析报告（时间占比）")
    print("="*100)
    print(f"总运行时间: {total_time:.4f} 秒")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*100)
    print(f"{'函数名':<60} {'累计时间(s)':<12} {'占比(%)':<10} {'调用次数':<10} {'平均时间(s)':<12}")
    print("-"*100)
    
    for func in report['functions']:
        print(f"{func['function'][:58]:<60} {func['cumulative_time']:<12.4f} "
              f"{func['percentage']:<10.2f} {func['call_count']:<10} "
              f"{func['average_time']:<12.4f}")
    
    # 保存报告为JSON
    with open('performance_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n详细报告已保存到 performance_report.json")
    return report

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    profiler = cProfile.Profile()
    profiler.enable()

    # ====== 你要分析的代码 ======
    main()
    # ===========================

    profiler.disable()
    profiler.dump_stats("profile.prof")
    report = generate_percentage_report("profile.prof")
    print(report)