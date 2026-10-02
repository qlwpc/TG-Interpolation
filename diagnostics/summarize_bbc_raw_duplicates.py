"""Validate and export the completed full-raw BBC duplicate census."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize(path):
    result = json.loads(path.read_text())
    assert result['complete'] and not result['pilot_rows_per_shard']
    sources = result['sources']
    assert len(sources) == 94 and all(s['complete_file'] for s in sources)
    profiles = result['profiles']
    all_rows = profiles['all_94_shards']
    historic = profiles['historical_89_shard_pool']
    reserved = profiles['reserved_5_shards']
    assert all_rows['documents'] == 15390361 == sum(s['rows'] for s in sources)
    assert historic['documents'] == 15323733 and reserved['documents'] == 66628
    for profile in profiles.values():
        hist = {int(k): v for k,v in profile['copies_per_unique_document_histogram'].items()}
        assert sum(hist.values()) == profile['unique_terminal_documents']
        assert sum(k*v for k,v in hist.items()) == profile['documents']
        assert profile['extra_duplicate_copies'] == profile['documents']-profile['unique_terminal_documents']
        assert profile['documents_in_duplicate_groups'] == profile['documents']-hist.get(1,0)
    overlap = result['reserved_overlap']
    assert all_rows['unique_terminal_documents'] == historic['unique_terminal_documents'] + reserved['unique_terminal_documents'] - overlap['reserved_unique_contents_with_historical_pool_match']
    assert reserved['documents'] == overlap['reserved_document_instances_with_historical_pool_match'] + overlap['reserved_document_instances_without_historical_pool_match']
    assert reserved['unique_terminal_documents'] == overlap['reserved_unique_contents_with_historical_pool_match'] + overlap['reserved_unique_contents_without_historical_pool_match']
    within = sum(s['within_shard_extra_copies'] for s in sources)
    cross_extra = sum(s['unique_within_shard'] for s in sources)-all_rows['unique_terminal_documents']
    assert within + cross_extra == all_rows['extra_duplicate_copies']
    root = path.parent
    with (root/'per_shard.csv').open('x',newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sources[0]))
        writer.writeheader();writer.writerows(sources)
    with (root/'copy_histograms.csv').open('x',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['scope','copies_per_content','content_groups','document_instances'])
        for name, profile in profiles.items():
            for copies, groups in profile['copies_per_unique_document_histogram'].items():
                writer.writerow([name,int(copies),groups,int(copies)*groups])
    compact = {k:v for k,v in result.items() if k not in ['sources','top_repeated_contents']}
    compact['within_shard_extra_copies'] = within
    compact['cross_shard_extra_distinct_appearances'] = cross_extra
    assert all_rows['unique_terminal_documents'] <= result['unique_raw_parse_lines'] <= all_rows['documents']
    compact['extra_identical_parsed_line_copies'] = all_rows['documents']-result['unique_raw_parse_lines']
    compact['identical_parsed_line_duplicate_percent'] = 100*compact['extra_identical_parsed_line_copies']/all_rows['documents']
    compact['reserved_actual_train_overlap_percent'] = 100*overlap['reserved_document_instances_with_actual_train_match']/reserved['documents']
    compact['reserved_historical_pool_overlap_percent'] = 100*overlap['reserved_document_instances_with_historical_pool_match']/reserved['documents']
    compact['all_historical_byte_checks_passed'] = all(result['validation'].values())
    compact['train_overlap_is_byte_verified'] = result['validation']['full_raw_to_historical_train_data_hash_equal']
    compact['train_membership_interpretation'] = (
        'Train-related counts use reconstructed historical row membership; '
        'if train_overlap_is_byte_verified is false, they are not a fresh measurement '
        'of the existing train.npy. Raw-pool duplicate statistics do not depend on this mapping.'
    )
    compact['aggregation_validation_passed'] = True
    compact['evidence_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [path,root/'manifest.json',root/'per_shard.csv',root/'copy_histograms.csv']}
    script = Path(__file__).with_name('audit_bbc_raw_duplicates.py')
    compact['census_script_sha256'] = hashlib.sha256(script.read_bytes()).hexdigest()
    with (root/'summary.json').open('x') as handle:
        json.dump(compact,handle,indent=2);handle.write('\n')
    return compact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('census_json',type=Path)
    args = parser.parse_args()
    result=summarize(args.census_json)
    for name, profile in result['profiles'].items():
        print(name,json.dumps({k:v for k,v in profile.items() if k!='copies_per_unique_document_histogram'}))
    print('validation',result['validation'])
    print('reserved_overlap',result['reserved_overlap'])
