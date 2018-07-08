from pprint import pprint
"""
    What is the vocab size?
    What's the max question length
    What's the max paragraph length.
"""

train_para = list(open('./data/squad/train.ids.context'))
train_ques = list(open('./data/squad/train.ids.question'))
test_para = list(open('./data/squad/val.ids.context'))
test_ques = list(open('./data/squad/val.ids.question'))

print len(train_para), len(train_ques), len(test_para), len(test_ques)

# max_vocab_id = 0
# for x in train_para+train_ques+test_para+test_ques:
#     max_vocab_id = max([int(id) for id in x.split()]) if max([int(id) for id in x.split()]) >= max_vocab_id else max_vocab_id
#
# print "Max vocab id: ", max_vocab_id            #115239
#
# unique_ids = []
# for x in train_para+train_ques+test_para+test_ques:
#     for id in x.split():
#         if not int(id) in unique_ids:
#             unique_ids.append(int(id))
#
# print "Total unique ids: ", len(unique_ids)     #115237
#


max_q_len = 0
q_len = {0: 0, 10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0}
for x in train_ques + test_ques:
    max_q_len = len(x.split()) if len(x.split()) > max_q_len else max_q_len
    q_len[(len(x.split()) / 10) * 10] += 1

print "max q len: ", max_q_len
pprint(q_len)

max_p_len = 0
p_len = {0: 0, 100: 0, 200: 0, 300: 0, 400: 0, 500: 0, 600: 0, 700: 0, 800: 0}
for x in train_para + test_para:
    max_p_len = len(x.split()) if len(x.split()) > max_p_len else max_p_len
    p_len[(len(x.split())/100)*100] += 1

print "max p len: ", max_p_len
pprint(p_len)