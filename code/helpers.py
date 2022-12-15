import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

def plot_bars(data, title, x, y, hue=None, size=(12,5), baseline=None, to_filename=None, fontsizes=None, legend=True):

    if fontsizes is None:
        fontsizes = {'title': 20, 'label': 12}
        
    f, ax = plt.subplots(1, figsize=size)
    
    # plot bars, set title and labels
    sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax)
    ax.set_title(title, fontsize=fontsizes['title'], pad=20)
    ax.xaxis.set_tick_params(labelsize=fontsizes['label'])
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_xlabel('')
    sns.despine(f)
    
    # plot baseline value
    if baseline is not None:
        ax.axhline(baseline, label='Baseline', color='red', linestyle='-.', alpha=0.7)
        ax.set_ylim(round(baseline / 5, 1), data[y].max()*1.25)
    
    # label bars
    for bar in ax.patches:
        bar_height = round(bar.get_height(), 3)
        ax.annotate(bar_height, (bar.get_x() + bar.get_width()/2, bar.get_height()*1.01), 
                    ha='center', color='black', fontsize=10)
    
    # set legend
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=13)

    f.tight_layout();
    
    # save to file
    if to_filename:
        f.savefig(f'../images/{to_filename}.png', transparent=True, bbox_inches="tight")



def eda_barplots(data, title, x, y, size=(12,5), xticklabels=None, to_filename=None, fontsizes=None):

    if fontsizes is None:
        fontsizes = {'title': 20, 'label': 12}
        
    f, ax = plt.subplots(1, figsize=size)
    
    # plot bars, set title and labels
    sns.barplot(x=x, y=y, data=data, ax=ax, ci=None)
    ax.set_title(title, fontsize=fontsizes['title'], pad=20)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    sns.despine(f)
    
    # label bars
    for bar in ax.patches:
        bar_height = round(bar.get_height(), 3)
        ax.annotate(bar_height, (bar.get_x() + bar.get_width()/2, bar.get_height()*1.01), 
                    ha='center', color='black', fontsize=10)
    
    f.tight_layout();
    
    # save to file
    if to_filename:
        f.savefig(f'../images/{to_filename}.png', transparent=True, bbox_inches="tight")


# using a given KFold split and a classifier, train and return the average f1-score
def cross_validation(X, y, kf, clfr, **kwargs):
    
    scores = []
    for train_idx, test_idx in kf.split(X, y):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        
        model = clfr(**kwargs).fit(X_train, y_train)
        sc = f1_score(y_test, model.predict(X_test))
        scores.append(sc)
    return sum(scores) / len(scores)