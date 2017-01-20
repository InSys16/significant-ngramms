# coding=utf-8
# from __future__ import division

import time
import codecs
import numpy as np

ENCODING = 'utf-8'


def dt2str(dt):
    mins = int(dt / 60)
    secs = int(dt) % 60
    return '{:>2} mins {:>2} secs'.format(mins, secs)


last_progress = -1
start_time = time.time()


def print_progress(cur, all, percent=20, force=False):
    global last_progress, start_time
    progress = int(cur * 100 / all)
    if force or (progress != last_progress and progress % percent == 0):
        last_progress = progress
        dt = time.time() - start_time
        print('{:>3}% processed... [{}]'.format(progress, dt2str(dt)))
        return True
    return False


def read_content(content_file_path, lines_to_read=-1):
    ru_lines = []
    line_count = 0
    line_count_mod = 1024

    print('Content file:', content_file_path)
    print('Reading content...')
    with codecs.open(content_file_path, encoding=ENCODING) as content_file:
        for line in content_file:
            ru_lines.append(line)
            line_count += 1
            if line_count % line_count_mod == 0:
                print('{:<7} lines processed...'.format(line_count))
                line_count_mod *= 2

            if line_count == lines_to_read:
                break

    print(line_count, 'lines processed in total.\n')

    return ru_lines


def split_into_sentences(lines, stopword_file, max_sentence_length=30):
    from nltk.stem.snowball import SnowballStemmer
    russian_letters = set(u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
    word_separators = set(" \t-%")
    sentence_separators = set(".!?")

    stemmer = SnowballStemmer('russian')

    stopwords = set()
    with codecs.open(stopword_file, encoding=ENCODING) as swf:
        for stop_word in swf:
            chars_only = stop_word[:-2]
            stopwords.add(stemmer.stem(chars_only.lower()))

    stem2num_dict = {}
    num2stem_dict = {}
    stem_count_dict = {}
    stem_counter = 0

    sentences = []
    groups = []
    stems = []

    line_count = 0
    total_line_count = len(lines)

    print('Splitting text into sentences...')

    for line in lines:
        st, i = -1, 0
        got_word = False

        for c in line:
            if c in russian_letters:  # got letter
                if st < 0:
                    st = i
            elif st >= 0:  # got char that is not a russian letter, word is over
                kkk = line[st:i]
                stemmed = stemmer.stem(kkk.lower())
                st = -1
                if stemmed not in stopwords:
                    if stemmed in stem2num_dict:
                        stem_num = stem2num_dict[stemmed]
                        stem_count_dict[stem_num] += 1
                    else:
                        stem_counter += 1
                        stem_num = stem_counter
                        stem2num_dict[stemmed] = stem_num
                        num2stem_dict[stem_num] = stemmed
                        stem_count_dict[stem_num] = 1

                    stems.append(stem_num)

                    if c not in word_separators:
                        groups.append(stems)
                        stems = []
                        if c in sentence_separators:
                            sentences.append(groups)
                            groups = []
                else:
                    if len(groups) > 0:
                        sentences.append(groups)
                        groups = []

            elif c in sentence_separators:
                if len(groups) > 0:
                    sentences.append(groups)
                    groups = []
            elif c not in word_separators:
                if len(stems) > 0:
                    groups.append(stems)
                    stems = []

            i += 1

        if len(stems) > 0:
            groups.append(stems)
            stems = []
        if len(groups) > 0:
            sentences.append(groups)
            groups = []

        if len(stems) > 0 or len(groups) > 0:
            print('stems', stems)
            print('groups', groups)

        line_count += 1
        print_progress(line_count, total_line_count)

    print(total_line_count, 'lines processed in total.\n')
    print('Sentences collected:   {}'.format(len(sentences)))
    print('Unique stem collected: {}'.format(stem_counter))

    min_len = 2
    max_len = max_sentence_length
    filtered_sentences = []
    bad_sentences_count = 0
    for sentence in sentences:
        sentence_length = sum([len(group) for group in sentence])
        if min_len <= sentence_length <= max_len:
            filtered_sentences.append(sentence)
        else:
            bad_sentences_count += 1

    print('Sentences in length limit:   {}'.format(len(filtered_sentences)))
    print()

    return num2stem_dict, stem_count_dict, filtered_sentences


def gen_basis(dim, basis_size=16):
    print('Generating basis...[basis size={}, dim={}]\n'.format(basis_size, dim))
    return np.random.random_integers(-1, 1, (basis_size, dim))


def get_sentence_hash(sentence, basis, basis_size):
    lsh = 0
    for i in range(basis_size):
        dot = sum([basis[i][stem_number] for group in sentence for stem_number in group])
        # (was <<= 1)
        lsh *= 2
        if np.sign(dot) > 0:
            # was lsh |= 1
            lsh += 1
    return lsh


def dedup_with_cosine(sentences, threshold):
    from math import sqrt

    bags = []
    filtered_sentences = []  # of indexes

    for sentence in sentences:
        sentence_stemm_nums2 = set([stemm_num for group in sentence for stemm_num in group])
        length_of_stemm_nums = len(sentence_stemm_nums2)

        is_dup = False
        for length, stemm_nums1 in bags:
            length_of_intersection = len(stemm_nums1 & sentence_stemm_nums2)
            cosine = length_of_intersection / sqrt(length * length_of_stemm_nums)
            if cosine >= threshold:
                is_dup = True
                break

        if not is_dup:
            bags.append((length_of_stemm_nums, sentence_stemm_nums2))
            filtered_sentences.append(sentence)

    return filtered_sentences


def deduplicate_with_basis(sentences, basis, basis_size, threshold):
    from multiprocessing import Pool
    from functools import partial

    sentences_count = len(sentences)
    progress_counter, progress_goal = 0, sentences_count

    buckets = {}

    print('Hashing sentences and splitting into buckets... [basis size = {}]'.format(basis_size))
    for i in range(sentences_count):
        sentence = sentences[i]
        hash_of_sentence = get_sentence_hash(sentence, basis, basis_size)
        if hash_of_sentence in buckets:
            buckets[hash_of_sentence].append(sentence)
        else:
            buckets[hash_of_sentence] = [sentence]
        progress_counter += 1
        print_progress(progress_counter, progress_goal, 20)

    print('{} sentences processed in total.\n'.format(sentences_count))

    filtered_sentences = []

    progress_counter, progress_goal = 0, len(buckets)
    print('Deduplicating sentences... [cosine with threshold = {}]'.format(threshold))

    cpu_cnt = 4
    pool = Pool(cpu_cnt)
    async_results = [pool.apply_async(dedup_with_cosine, (sentences, threshold)) for hash, sentences in list(buckets.items())]

    pool.close()
    pool.join()
    filtered_sentences = [sentence for async_result in async_results for sentence in async_result.get()]

    initial_count = sentences_count
    filtered_count = len(filtered_sentences)
    thrown_away = initial_count - filtered_count
    filtered_percent = int((100 * thrown_away) / initial_count)

    print('{}% filtered, {}/{} sentences left.\n'.format(filtered_percent, filtered_count, initial_count))

    return filtered_sentences


def deduplicate_sentences(sentences, dim, basis_size=16, threshold=0.9):
    basis = gen_basis(dim, basis_size)
    return deduplicate_with_basis(sentences, basis, basis_size, threshold)


def sentences_to_groups(sentences):
    print('Splitting sentences into groups...')
    groups = [group for sentence in sentences for group in sentence]
    print('{} group in total.\n'.format(len(groups)))
    return groups


def count_ngrams(groups, n=2):
    ns = range(2, n + 1)
    ngram_dict = dict()
    i, count = 0, len(groups)

    print('Counting ngrams...')
    for group in groups:
        size_of_group = len(group)
        for j in range(size_of_group):
            for k in ns:
                if j + k <= size_of_group:
                    key = tuple(group[j:j + k])
                    if key not in ngram_dict:
                        ngram_dict[key] = 1
                    else:
                        ngram_dict[key] += 1
        i += 1
        print_progress(i, count)

    print(count, ' groups processed in total.\n')
    return ngram_dict


def count_ngram_words(ngram_dict, n=2):
    ns = range(2, n + 1)
    word_count_dict = dict()
    i, count = 0, len(ngram_dict)

    print('Counting ngram words...')
    for ngramm, count_of_ngramm in list(ngram_dict.items()):
        len_of_ngramm = len(ngramm)
        for position_in_ngramm in range(len(ngramm)):
            key = ngramm[position_in_ngramm], position_in_ngramm, len_of_ngramm
            if key in word_count_dict:
                word_count_dict[key] += count_of_ngramm
            else:
                word_count_dict[key] = count_of_ngramm
        i += 1
        print_progress(i, count)

    print(count, 'ngrams processed in total.\n')
    return word_count_dict


def rank_ngrams(ngram_dict, word_count, n=2, topn=10000):
    import math
    ns = range(2, n + 1)
    ranked_lists = {k: [] for k in ns}
    i, count = 0, len(ngram_dict)

    print('Ranking ngrams...')
    for ngramm, count_of_ngramm in list(ngram_dict.items()):
        len_of_ngramm = len(ngramm)
        rk = math.log(count_of_ngramm)
        for count in [word_count[(ngramm[position_in_ngramm], position_in_ngramm, len_of_ngramm)] for position_in_ngramm in range(len_of_ngramm)]:
            rk *= count_of_ngramm / count
        ranked_lists[len_of_ngramm].append((rk, ngramm))
        i += 1
        print_progress(i, count)

    print(count, 'ngrams processed in total.\n')

    for len_of_ngramm in ns:
        print('{}-grams collected: {}'.format(len_of_ngramm, len(ranked_lists[len_of_ngramm])))

    print()
    print('Sorting by rank...')
    for len_of_ngramm in ns:
        count = len(ranked_lists[len_of_ngramm])
        print('{}-grams collected: {}\nSorting...'.format(len_of_ngramm, count))
        ranked_lists[len_of_ngramm].sort(key=lambda x: x[0], reverse=True)
        print('Keeping top-{}'.format(topn))
        print()

        ranked_lists[len_of_ngramm] = ranked_lists[len_of_ngramm][:topn]

    print('Finished sorting.\n')
    return ranked_lists


def print_results(sorted_ranked_lists, word_dict, output_file_postfix, n, topn=0):
    ns = range(2, n + 1)
    for k in ns:
        out_file_name = '%s-grams-%s.txt' % (k, output_file_postfix)
        i, count = 0, len(sorted_ranked_lists[k])
        top = []
        print('Printing results...')
        with codecs.open(out_file_name, 'w+', encoding=ENCODING) as out:
            for rk, wis in sorted_ranked_lists[k]:
                w = u' '.join([word_dict[wi] for wi in wis])
                s = u'{:<40} {}\n'.format(w, rk)
                out.write(s)
                if i < topn:
                    top.append(s.encode(ENCODING))
                i += 1
                print_progress(i, count)

        print(count, 'ngrams processed in total.\n')


def find_significant_ngrams(content_file_path, stopword_file, output_file_postfix, n=2):
    print('Algorithm for finding significant ngrams in text')
    ru_lines = read_content(content_file_path)
    num2stem_dict, stem_count_dict, sentences = split_into_sentences(ru_lines, stopword_file)
    del ru_lines
    dim = len(num2stem_dict) + 1
    sentences = deduplicate_sentences(sentences, dim, basis_size=16, threshold=0.95)
    groups = sentences_to_groups(sentences)
    del sentences
    ngram_dict = count_ngrams(groups, n)
    del groups
    word_count_dict = count_ngram_words(ngram_dict, n)
    sorted_ranked_lists = rank_ngrams(ngram_dict, word_count_dict, n, topn=1000)
    del ngram_dict
    del word_count_dict
    print_results(sorted_ranked_lists, num2stem_dict, output_file_postfix, n, topn=50)
    print('Successfully finished!')


if __name__ == "__main__":
    import sys, time

    arg_cnt = 5
    if len(sys.argv) < arg_cnt:
        print('Usage %s "content_file" "stop_word_file" "output_file_postfix" N' % sys.argv[0])
        exit(1)

    find_significant_ngrams(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))