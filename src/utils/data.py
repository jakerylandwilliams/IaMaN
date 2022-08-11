import re, json, os
import numpy as np
from collections import defaultdict

def load_wikitext(v = "2", split = "train", num_articles = 10, seed = 691, all_articles = None, rebuild = False, space = True):
    if all_articles is not None:
        articles = np.array(all_articles)
    else:
        file_path = '/local-data/wikitext-' + str(v) + '/wiki.' + split + '.articles.json'
        if os.path.exists(file_path) and not rebuild:
            articles = json.load(open(file_path))
        else:
            sec_title, sec_level = '', 0
            articles = []; article = {'title': '', 'text': '', 'document': [], 'sections': []}
            for line in open('/local-data/wikitext-' + str(v) + '/wiki.' + split + '.tokens'):
                if re.search("^( =(?: =)* )([^=]+)(?: =(?: =)* )$", line):
                    level, title_text = re.search("^( =(?: =)* )([^=]+)(?: =(?: =)* )$", line).groups()
                    level = len(re.split("=", level)) - 1; title_text = title_text.strip()
                    if level == 1:
                        if article['title']:
                            articles.append(article)
                        sec_title, sec_level = '', 0
                        article = {'title': title_text.strip(), 'text': '', 'document': [], 'sections': []}
                    else:
                        sec_title = title_text
                        sec_level = level
                else:
                    if article['title'] and level and sec_title:
                        line = line.strip(" ")
                        article['text'] += line
                        if space:
                            article['document'].append([t for t in re.split("( )", line) if t])
                        else:
                            article['document'].append([t for t in re.split("([^ ]+| +[^ ]+)", line) if t])
                        article['sections'].append([sec_level, sec_title])
            if article['title']:
                articles.append(article)

            with open(file_path, "w") as f:
                f.write(json.dumps(articles))
            
    if num_articles:
        np.random.seed(seed)
        articles_sample = np.random.choice(articles, size=num_articles, replace=False)
    else:
        articles_sample = np.array(articles)
            
    return articles_sample

def load_ud(language, num_articles = 10, seed = 691, all_articles = None, rebuild = False, load_set = 'all', space = True):
    available_languages = defaultdict(list)
    for ud_dir in os.listdir('/local-data/UD/ud-treebanks-v2.9/'):
        if ud_dir[:3] == "UD_":
            lang, set_name = ud_dir[3:].split('-')
            available_languages[lang].append(set_name)
    assert(language in available_languages)
    if load_set == 'all':
        load_sets = list(available_languages[language])
    else: 
        assert(load_set in available_languages[language])
        load_sets = list([load_set])
    
    if all_articles is not None:
        articles = np.array(all_articles)
    else:
        articles = []; extra_space = tuple()
        for set_name in load_sets:
            filenames = [x for x in os.listdir(f"/local-data/UD/ud-treebanks-v2.9/UD_{language}-{set_name}/") if 'conllu' in x]
            for filename in filenames:
                file_path = (f"/local-data/UD/ud-treebanks-v2.9/UD_{language}-{set_name}/" + 
                             re.sub(".conllu", "-articles.json", filename)) 
                if os.path.exists(file_path) and not rebuild:
                    path_articles = json.load(open(file_path))
                else:
                    path_articles = []; doc_id = ''; extra_space = tuple()
                    article = {'id': '', 'title': '', 'text': '',
                               'document': [], 'conllu': [], 's_type': []}
                    idx_map = []; contracted_range = []; contraction_form = ""
                    for line in open(f"/local-data/UD/ud-treebanks-v2.9/UD_{language}-{set_name}/" + filename):
                        line = line.strip()
                        if re.search("^# newdoc id = ", line):
                            doc_id = line[14:]+"-"+set_name+"-"+filename[:-7]
                            if article['id']:
                                article['document'] = [list(s[:-1]+[s[-1]+" "]) for s in article['document']]
                                article['text'] = [t for s in article['document'] for t in s]
                                article['text'] = ''.join(article['text'])
                                newconllu = []
                                for s_i in range(len(article['conllu'])):
                                    newconllu.append([])
                                    for wi in range(len(article['conllu'][s_i])):
                                        newrow = list(article['conllu'][s_i][wi])
                                        newrow[0] = idx_map[s_i][article['conllu'][s_i][wi][0]]
                                        newrow[6] = idx_map[s_i][article['conllu'][s_i][wi][6]]
                                        newrow[8] = re.sub("^\d+", idx_map[s_i][article['conllu'][s_i][wi][6]], 
                                                           article['conllu'][s_i][wi][8])
                                        newconllu[-1].append(newrow)
                                    if space:
                                        newconllu[-1].append([str(int(newconllu[-1][-1][0])+1), ' ', ' ', 'SPACE', 
                                                              ' ', '_', newconllu[-1][-1][0], 'space', 
                                                              newconllu[-1][-1][0]+':space', '_'])
                                    else:
                                        newconllu[-1][-1][1] = newconllu[-1][-1][1] + ' '
                                    # if not space:
                                    #     newconllu[-1][0][1] = newconllu[-1][0][1].lstrip(' ')
                                    article['text'] = article['text'] + ' '
                                article['conllu'] = list(newconllu)
                                path_articles.append(article)
                            extra_space = tuple()
                            article = {'id': doc_id,
                                       'title': '', 'text': '', 
                                       'document': [], 'conllu': [], 's_type': []}
                            idx_map = []
                        if re.search("^# meta::title = ", line) and doc_id:
                            article['title'] = line[16:]
                        if re.search("^# sent_id = ", line):
                            if not doc_id:
                                if article['id']:
                                    article['document'] = [list(s[:-1]+[s[-1]+" "]) for s in article['document']]
                                    article['text'] = [t for s in article['document'] for t in s]
                                    article['text'] = ''.join(article['text'])
                                    newconllu = []
                                    for s_i in range(len(article['conllu'])):
                                        newconllu.append([])
                                        for wi in range(len(article['conllu'][s_i])):
                                            newrow = list(article['conllu'][s_i][wi])
                                            newrow[0] = idx_map[s_i][article['conllu'][s_i][wi][0]]
                                            newrow[6] = idx_map[s_i][article['conllu'][s_i][wi][6]]
                                            newrow[8] = re.sub("^\d+", idx_map[s_i][article['conllu'][s_i][wi][6]], 
                                                               article['conllu'][s_i][wi][8])
                                            newconllu[-1].append(newrow)
                                        if space:
                                            newconllu[-1].append([str(int(newconllu[-1][-1][0])+1), ' ', ' ', 'SPACE', 
                                                                  ' ', '_', newconllu[-1][-1][0], 'space', 
                                                                  newconllu[-1][-1][0]+':space', '_'])
                                        else:
                                            newconllu[-1][-1][1] = newconllu[-1][-1][1] + ' '
                                    # if not space:
                                    #     newconllu[-1][0][1] = newconllu[-1][0][1].lstrip(' ')
                                    article['conllu'] = list(newconllu)
                                    path_articles.append(article)
                                article = {'id': line[11:]+"-"+set_name+"-"+filename[:-7], 
                                           'title': '', 'text': '', 
                                           'document': [], 'conllu': [], 's_type': []}
                                idx_map = []
                            extra_space = tuple()
                            article['document'].append([])
                            article['conllu'].append([])
                            idx_map.append({'0': '0'})
                        if re.search("^# s_type = ", line):
                            article['s_type'].append(line[11:])
                        if re.search("^# text = ", line):
                            article['text']+= line[9:]+" " # + "\n"
                        if re.search("^(\d+)\t([^\t]+)\t", line):
                            row = re.split("\t", line)
                            if contraction_form:
                                if contracted_range[-1] == int(row[0]):
                                    constituent_chunk = contraction_form
                                    contraction_form = ""
                                    contracted_range = []
                                else:
                                    constituent_chunk = ""
                                    for chix, ch in enumerate(contraction_form):
                                        if ch in row[1]:
                                            if len(constituent_chunk):
                                                if row[1].index(ch) <= row[1].index(constituent_chunk[-1]):
                                                    break
                                            constituent_chunk += ch
                                        elif ch == "'" or ch == "’":
                                            constituent_chunk += ch
                                        else:
                                            break
                                        if len(constituent_chunk) == len(row[1]): break
                                    contraction_form = contraction_form[len(constituent_chunk):]
                                row[1] = constituent_chunk
                            if extra_space and space:
                                idx_map[-1][extra_space[0][0]] = str(len(idx_map[-1]))
                                article['conllu'][-1].append(extra_space[0])
                                article['document'][-1].append(extra_space[1])
                                extra_space = tuple()
                            elif (not article['document'][-1]) and space: # this one prepends the sentence with a space
                                extra_space = (["0.1", " ", " ", "SPACE", " ", "_", "1", "space", "1:space", "_"], " ")
                                idx_map[-1]['0.1'] = str(len(idx_map[-1]))
                                article['conllu'][-1].append(extra_space[0])
                                article['document'][-1].append(extra_space[1])
                                extra_space = tuple()
                            elif space:
                                extra_space = tuple()
                            idx_map[-1][row[0]] = str(len(idx_map[-1]))
                            article['conllu'][-1].append(row)
                            article['document'][-1].append(row[1])
                            if extra_space and not space:
                                article['conllu'][-1][-1][1] = " " + article['conllu'][-1][-1][1]
                                article['document'][-1][-1] = " " + article['document'][-1][-1]
                                extra_space = tuple()
                            if 'SpaceAfter=No' not in row[-1] and not contraction_form:
                                extra_space = ([row[0]+".1", " ", " ", "SPACE", " ", "_", str(int(row[0])+1), "space", 
                                                row[0]+":space", "_"], " ")
                        elif re.search("^(\d+-\d+)\t([^\t]+)\t", line):
                            row = re.split("\t", line)
                            contracted_range = list(map(int, row[0].split('-')))
                            contraction_form = row[1]
                        else:
                            contracted_range = []
                            contraction_form = ""
                            
                    if article['id']:
                        article['document'] = [list(s[:-1]+[s[-1]+" "]) for s in article['document']]
                        article['text'] = [t for s in article['document'] for t in s]
                        article['text'] = ''.join(article['text'])
                        newconllu = []
                        for s_i in range(len(article['conllu'])):
                            newconllu.append([])
                            for wi in range(len(article['conllu'][s_i])):
                                newrow = list(article['conllu'][s_i][wi])
                                newrow[0] = idx_map[s_i][article['conllu'][s_i][wi][0]]
                                newrow[6] = idx_map[s_i][article['conllu'][s_i][wi][6]]
                                newrow[8] = re.sub("^\d+", idx_map[s_i][article['conllu'][s_i][wi][6]], 
                                                   article['conllu'][s_i][wi][8])
                                newconllu[-1].append(newrow)
                            if space:
                                newconllu[-1].append([str(int(newconllu[-1][-1][0])+1), ' ', ' ', 'SPACE', 
                                                      ' ', '_', newconllu[-1][-1][0], 'space', 
                                                      newconllu[-1][-1][0]+':space', '_'])
                            else:
                                newconllu[-1][-1][1] = newconllu[-1][-1][1] + ' '
                        # if not space:
                        #     newconllu[-1][0][1] = newconllu[-1][0][1].lstrip(' ')
                        article['conllu'] = list(newconllu)
                        path_articles += [article]
                    with open(file_path, "w") as f:
                        f.write(json.dumps(path_articles))
                if path_articles:
                    articles += path_articles
    if num_articles:
        np.random.seed(seed)
        articles_sample = np.random.choice(articles, size=num_articles, replace=False)
    else:
        articles_sample = np.array(articles)
            
    return articles_sample