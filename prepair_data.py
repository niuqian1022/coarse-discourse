import pandas as pd

comments = []
labels = []

label_categories = ["question", "answer", "announcement",
                    "appreciation","agreement", "elaboration",
                    "disagreement", "humor", "negativereaction"]

label_map = dict((s, i) for i, s in enumerate(label_categories))

with open('coarse_discourse_dump_reddit.json', 'r') as f:
    for line_idx, line in enumerate(f):
        if line_idx >= 9482: # last line is incomplete
            break

        df = pd.read_json(line.encode('utf-8').strip())
        crt_post_df = df["posts"]
        for post_idx in range(len(crt_post_df)):
            annotation_df = crt_post_df[post_idx]["annotations"]
            # check if all annotation agrees
            agree = len(set(annotation_df[i]["main_type"] for i in range(len(annotation_df)))) == 1
            if agree:
                if "body" in crt_post_df[post_idx] and annotation_df[0]["main_type"] in label_map:
                    comments.append(crt_post_df[post_idx]["body"].encode('utf-8'))
                    labels.append(label_map[annotation_df[0]["main_type"]])
        if line_idx % 1000 == 0:
            print('.')

assert len(comments)==len(labels), "number of comments and labels should match"

# random split train and test
perm_idx = np.random.permutation(len(comments))
ntrain = int(0.8*len(comments))
ntest = len(comments) - ntrain
train_idx = perm_idx[:ntrain]
test_idx = perm_idx[ntrain:]

with open('train_comments.csv', 'w') as f:
    for idx in train_idx:
        c = comments[idx]
        c = c.replace('\n', ' ').strip()
        f.write("{},{}\n".format(c, labels[idx]))

with open('test_comments.csv', 'w') as f:
    for idx in test_idx:
        c = comments[idx]
        c = c.replace('\n', ' ').strip()
        f.write("{},{}\n".format(c, labels[idx]))