import re, json, os
import numpy as np

def load_wikitext(v = "2", split = "train", num_articles = 10, seed = 691, all_articles = None):
    if all_articles is not None:
        articles = np.array(all_articles)
    else:
        file_path = '/data/WikiText/wikitext-' + str(v) + '/wiki.' + split + '.articles.json'
        if os.path.exists(file_path):
            articles = json.load(open(file_path))
        else:
            sec_title, sec_level = '', 0
            articles = []; article = {'title': '', 'text': '', 'document': [], 'sections': []}
            for line in open('/data/WikiText/wikitext-' + str(v) + '/wiki.' + split + '.tokens'):
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
                        article['text'] += line
                        article['document'].append([t for t in line.split(" ") if t])
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

def load_gum(num_articles = 10, seed = 691, all_articles = None, rebuild = False):
    if all_articles is not None:
        articles = np.array(all_articles)
    else:
        filenames = [x for x in os.listdir('/data/gum/gum-master/dep/') if 'conllu' in x]
        articles = []
        for filename in filenames:
            
            file_path = '/data/gum/gum-master/dep/' + re.sub(".conllu", "-articles.json", filename) 
            if os.path.exists(file_path) and not rebuild:
                path_articles = json.load(open(file_path))
            else:
                path_articles = []
                article = {'id': '', 'title': '', 'text': '',
                           'document': [], 'conllu': []}
                parent_map = []
                for line in open('/data/gum/gum-master/dep/' + filename):
                    line = line.strip()
                    if re.search("^# newdoc id = ", line):
                        if article['id']:
                            #for s_ix in range(len(article['conllu'])):
                            #    for t_ix in range(len(article['conllu'][s_ix])):
                            #        article['conllu'][s_ix][t_ix][0] = str(parent_map[s_ix][article['conllu'][s_ix][t_ix][0]])
                            #        article['conllu'][s_ix][t_ix][-4] = str(parent_map[s_ix][article['conllu'][s_ix][t_ix][-4]])
                            #        article['conllu'][s_ix][t_ix][-2] = "|".join([
                            #            str(parent_map[s_ix][dep[:dep.index(":")]])+":"+dep[dep.index(":"):]
                            #            for dep in article['conllu'][s_ix][t_ix][-2].split("|")])
                            path_articles.append(article)
                        article = {'id': line[14:],
                                   'title': '', 'text': '', 
                                   'document': [], 'conllu': []}
                        parent_map = []
                    if re.search("^# meta::title = ", line):
                        article['title'] = line[16:]
                    if re.search("^# sent_id = ", line):    
                        article['document'].append([])
                        article['conllu'].append([])
                        parent_map.append({})
                    if re.search("^# text = ", line):
                        article['text'] += line[9:] + "\n"
                    if re.search("^(\d+)\t([^\t]+)\t", line):
                        row = re.split("\t", line)
                        parent_map[-1][row[0]] = len(parent_map) + 1
                        article['conllu'][-1].append(row)
                        article['document'][-1].append(row[1])
                        if 'SpaceAfter=No' not in row[-1]:
                            parent_map[-1][row[0]+".1"] = parent_map[-1][row[0]]+1
                            article['conllu'][-1].append([row[0]+".1", " ", " ", "PUNCT", 
                                                          " ", "_", row[0], "punct", 
                                                          row[0]+":punct", "_"])
                            article['document'][-1].append(" ")
                if article['id']:
                    #for s_ix in range(len(article['conllu'])):
                    #    for t_ix in range(len(article['conllu'][s_ix])):
                    #        print(article['conllu'][s_ix])
                    #        article['conllu'][s_ix][t_ix][0] = str(parent_map[s_ix][article['conllu'][s_ix][t_ix][0]])
                    #        article['conllu'][s_ix][t_ix][-4] = str(parent_map[s_ix][article['conllu'][s_ix][t_ix][-4]])
                    #        article['conllu'][s_ix][t_ix][-2] = "|".join([
                    #            str(parent_map[s_ix][dep[:dep.index(":")]])+":"+dep[dep.index(":"):]
                    #            for dep in article['conllu'][s_ix][t_ix][-2].split("|")])
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