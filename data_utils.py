import re

def tokenize(sent):
    stop_words = {"a", "an", "the"}
    sent = sent.lower()
    if sent == '<silence>':
        return [sent]
    result = [word.strip() for word in re.split('(\W+)?', sent) 
              if word.strip() and word.strip() not in stop_words]
    if not result:
        result = ['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result = result[:-1]
    return result

def parse_dialogs_per_response(lines, candidates_to_idx):
    data = []
    facts_temp = []
    utterance_temp = None
    response_temp = None
    for line in lines:
        line = line.strip()
        if line:
            nid, line = line.split(' ', 1)
            if '\t' in line: 
                utterance_temp, response_temp = line.split('\t')
                answer = candidates_to_idx[response_temp]
                utterance_temp = tokenize(utterance_temp)
                response_temp = tokenize(response_temp)
                data.append((facts_temp[:], utterance_temp[:], answer))
                utterance_temp.append('$u')
                response_temp.append('$r')
                utterance_temp.append('#' + nid)
                response_temp.append('#' + nid)
                facts_temp.append(utterance_temp)
                facts_temp.append(response_temp)
            else: 
                response_temp = tokenize(line)
                response_temp.append('$r')
                response_temp.append('#' + nid)
                facts_temp.append(response_temp)
        else: 
            facts_temp = []
    return data
