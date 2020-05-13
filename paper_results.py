import pandas as pd
from scipy.stats.stats import spearmanr
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from statsmodels.formula.api import logit
from collections import defaultdict
from corrstats import dependent_corr
import pickle
import numpy as np
from scipy import spatial
import os

def main():
    os.makedirs('plots', exist_ok=True)
    _, cat_to_ent, ent_to_cat, cat_to_class, class_to_cat = read_dataset('exp2_instantiation/modified_dataset/Pos+Inst2Inst.txt')
    polysemy_data = pd.read_csv('auxiliary_data/polysemy_data.csv', index_col="category")
    analysis_for_exp1(cat_to_ent, ent_to_cat, cat_to_class, class_to_cat, polysemy_data)
    analysis_for_exp2(cat_to_ent, ent_to_cat, cat_to_class, class_to_cat, polysemy_data)


def analysis_for_exp1(cat_to_ent, ent_to_cat, cat_to_class, class_to_cat, polysemy_data):
    print("Loading data for Experiment 1")
    exp1_results = pd.read_csv('exp1_relatedness/exp1_aggregate_scores.csv')    # Rename this to exp1 results
    for i, row in exp1_results.iterrows():
        exp1_results.at[i, 'num_synsets'] = (polysemy_data.at[row['cat1'], 'num_synsets'] + polysemy_data.at[row['cat2'], 'num_synsets']) / 2
        exp1_results.at[i, 'clear'] = polysemy_data.at[row['cat1'], 'clear'] & polysemy_data.at[row['cat2'], 'clear']
        exp1_results.at[i, 'cat1_domain'] = cat_to_class[row['cat1']]
        exp1_results.at[i, 'cat2_domain'] = cat_to_class[row['cat2']]
    exp1_results['within'] = exp1_results['cat1_domain'] == exp1_results['cat2_domain']

    # # Merge domains
    # master_per_pair['cat1_domain'].replace(to_replace={'communication': 'Other', 'artifact': 'Other', 'act': 'Other'}, inplace=True)
    # master_per_pair['cat2_domain'].replace(to_replace={'communication': 'Other', 'artifact': 'Other', 'act': 'Other'}, inplace=True)

    print("---------------------------------")


    #########################################################
    ## Table 1: Compositon of the instantiation dataset >5 ##
    #########################################################
    print("Table 1: Composition of the instantiation dataset >5 ")
    print(f"categories:  {len(cat_to_ent)}, instantiations: {sum([len(cat_to_ent[cat]) for cat in cat_to_ent])}, entities: {len(ent_to_cat)}")
    for class_ in class_to_cat:
        print(f"{class_.upper()}:  categories:  {len(class_to_cat[class_])}, instantiations: {len([ent for cat in class_to_cat[class_] for ent in cat_to_ent[cat]])}, entities: {len(set([ent for cat in class_to_cat[class_] for ent in cat_to_ent[cat]]))}")
        print('  ', class_to_cat[class_])
    print("---------------------------------")

    #############################################
    ## Table 2: counts per domain Experiment 1 ##
    #############################################
    print("Table 2. Counts per domain Experiment 1:")
    print(exp1_results.groupby(['cat1_domain', 'within'])['human_relatedness'].count())
    print(exp1_results.groupby(['cat2_domain', 'within'])['human_relatedness'].count())
    print("---------------------------------")

    #######################################
    ## Figure 2: Histograms experiment 1 ##
    #######################################
    plt.rcParams.update({'font.size': 14})
    f, ax = plt.subplots(1,3,figsize=(15,5),sharey=True,sharex=True)
    ax[0].hist(x=[exp1_results.loc[~exp1_results['within']].human_relatedness, exp1_results.loc[exp1_results['within']]['human_relatedness']], stacked=True)
    ax[1].hist(x=[exp1_results.loc[~exp1_results['within']].nounbased_cos, exp1_results.loc[exp1_results['within']]['nounbased_cos']], stacked=True)
    ax[2].hist(x=[exp1_results.loc[~exp1_results['within']].namebased_cos, exp1_results.loc[exp1_results['within']]['nounbased_cos']], stacked=True, label=["between-domain", "within-domain"])
    ax[0].title.set_text("Human relatedness scores")
    ax[1].title.set_text("Noun-based model (cos)")
    ax[2].title.set_text("Name-based model (cos)")
    plt.legend()
    # plt.show()
    path = 'plots/exp1_histograms.png'
    plt.savefig(path)
    print(f'Histograms saved to: {path}')

    print('------------------------')

    #################################
    ####### SCATTERPLOTS EXP 1 ######
    #################################

    # Colored by within/between:
    exp1_results['Subset'] = exp1_results['within'].apply(lambda x: "Within-domain" if x else "Between-domain")
    for model, ylabel in [('nounbased_cos', 'Noun-based model'), ('namebased_cos', 'Name-based model')]:
        plt.close()
        palette = sns.color_palette()[0:2]
        ax = sns.scatterplot(data=exp1_results, hue='Subset', hue_order=['Between-domain', 'Within-domain'], x='human_relatedness', y=model, zorder=10, palette=palette)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlabel('Human judgments', fontsize=16)
        plt.ylim(-0.2,1)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12)
        plt.tight_layout()
        # plt.show()
        path = f'plots/exp1_scatterplot_{model}-overall.png'
        plt.savefig(path)
        print(f'Scatterplot saved to: {path}')

    print('--------------')

    ##################################
    ### Table 3: Main results exp 1 ##
    ##################################
    print("Table 3: Main results exp 1.")
    for df, title in zip(
            [exp1_results, exp1_results.loc[exp1_results['clear']], exp1_results.loc[~exp1_results['clear']], exp1_results.loc[exp1_results['within']], exp1_results.loc[~exp1_results['within']]],
            ['all', 'clear', 'unclear', 'within', 'between']
    ):
        print(title)
        print("  Number of pairs: ", len(df))
        spearman_predicate_human, p = spearmanr(df['nounbased_cos'], df['human_relatedness'])
        print("  - NounBased:", spearman_predicate_human, p)
        spearman_centroid_human, p = spearmanr(df['namebased_cos'], df['human_relatedness'])
        print("  - NameBased:", spearman_centroid_human, p)
        spearman_centroid_predicate, p = spearmanr(df['namebased_cos'], df['nounbased_cos'])
        n_datapoints = len(df)
        t, pval = dependent_corr(spearman_centroid_human, spearman_predicate_human, spearman_centroid_predicate, n_datapoints)
        print("  Steiger's dependent correlations test:", t, pval)

    print('------------------------')

    ##############################################
    ### Table 4: Exp 1 correlation with synsets ##
    ##############################################
    print("Table 4: Exp1 effect of polysemy.")
    catp2judgs = {}
    catp2preds_nounbased = {}
    catp2preds_namebased = {}
    for i, row in exp1_results.iterrows():
        catp2judgs[(row['cat1'], row['cat2'])] = row['human_relatedness']
        catp2preds_nounbased[(row['cat1'], row['cat2'])] = row['nounbased_cos']
        catp2preds_namebased[(row['cat1'], row['cat2'])] = row['namebased_cos']

    rank_delta_nounbased = rank_delta(list(catp2judgs.keys()), catp2judgs, catp2preds_nounbased)
    rank_delta_namebased = rank_delta(list(catp2judgs.keys()), catp2judgs, catp2preds_namebased)

    for i, row in exp1_results.iterrows():
        exp1_results.at[i, 'delta_rank_nounbased'] = rank_delta_nounbased[(row['cat1'], row['cat2'])]
        exp1_results.at[i, 'delta_rank_namebased'] = rank_delta_namebased[(row['cat1'], row['cat2'])]

    for df, title in zip(
            [exp1_results, exp1_results.loc[exp1_results['clear']], exp1_results.loc[~exp1_results['clear']], exp1_results.loc[exp1_results['within']], exp1_results.loc[~exp1_results['within']]],
            ['all', 'clear', 'unclear', 'within', 'between']
    ):
        print(title)
        print("  Number of pairs: ", len(df))

        rho_nounbased, p = spearmanr(df['delta_rank_nounbased'].apply(abs), df['num_synsets'])
        print("  - NounBased:", rho_nounbased, p)

        rho_namebased, p = spearmanr(df['delta_rank_namebased'].apply(abs), df['num_synsets'])
        print("  - NameBased:", rho_namebased, p)

        rho_nounnamebased, p = spearmanr(df['delta_rank_namebased'].apply(abs), df['delta_rank_nounbased'].apply(abs))
        n_datapoints = len(df)
        t, pval = dependent_corr(rho_namebased, rho_nounbased, rho_nounnamebased, n_datapoints)
        print("  Steiger's dependent correlations test:", t, pval)

    print('------------------------------')


def analysis_for_exp2(cat_to_ent, ent_to_cat, cat_to_class, class_to_cat, polysemy_data):
    ##############################
    ## Experiment 2 preparation ##
    ##############################
    print("Computing data for Experiment 2.")
    exp2_results_files = [
        ('Inverse', 'exp2_predictions_Inverse.csv'),
        ('Inst2Inst', 'exp2_predictions_Inst2Inst.csv'),
        ('NotInst_global', 'exp2_predictions_NotInst_global.csv'),
        ('NotInst_inDomain', 'exp2_predictions_NotInst_inDomain.csv'),
        ('Union_global', 'exp2_predictions_Union_global.csv'),
        ('Union_inDomain', 'exp2_predictions_Union_inDomain.csv'),
    ]
    f1_per_category = {}
    results_exp2_master = []
    for partition, path in exp2_results_files:
        f1_per_category[partition] = {}
        exp2_results = pd.read_csv('exp2_instantiation/' + path)
        exp2_results['category'] = exp2_results.apply(lambda x: x['element1'] if x['type'] in ['Inverse'] else x['element2'] if x['type'] in ["Pos", "NotInst_global", "NotInst_inDomain"] else None, axis=1)
        exp2_results['clear'] = exp2_results['category'].apply(lambda x: polysemy_data.loc[x, 'clear'] if x is not None else None)

        for subset, df in [
            ('all', exp2_results),
            ('all*', exp2_results.loc[~exp2_results['category'].isna()].copy()),
            ('clear', exp2_results.loc[exp2_results['clear'] == True]),
            ('unclear', exp2_results.loc[exp2_results['clear'] == False]),
        ]:
            f1_per_category[partition][subset] = {}
            n_unique_categories = len(df.loc[~df['category'].isna()]['category'].unique())
            proportion_pos = sum(df['type'] == 'Pos') / len(df)
            for model in ['baseline_freq', 'baseline_pos', 'nounbased_cos', 'nounbased_1hl', 'nounbased_2hl', 'namebased_cos', 'namebased_1hl', 'namebased_2hl']:
                model_prefix, model_suffix = model.split('_')
                f1_per_cat = df.loc[~df['category'].isna()].groupby('category').apply(lambda x: f1_score(x['gold'], x[model], pos_label="Yes"))
                f1_per_category[partition][subset][model] = f1_per_cat.to_dict()
                results_exp2_master.append([partition, subset, model_prefix, model_suffix, f1_score(df['gold'], df[model], pos_label="Yes"),
                                            f1_per_cat.mean(),
                                            len(df),
                                            n_unique_categories,
                                            proportion_pos,
                                            ])
    results_exp2_master = pd.DataFrame(results_exp2_master, columns=['type', 'subset', 'model', 'model_spec', 'f1', 'f1_per_cat', 'num_datapoints', 'num_cats', 'proportion_pos'])
    results_exp2_master.set_index(['type', 'subset', 'model', 'model_spec'], inplace=True)

    print('-----------------------------')

    ##################################################
    ## Table 7: Exp 2 F1 scores all models/datasets ##
    ##################################################
    # pd.options.display.float_format = '{:,.2f}'.format

    print("Table 7: Exp 2 F1 scores all models/datasets")
    print(results_exp2_master.loc[(slice(None), 'all', slice(None), slice(None)), :].reorder_levels(
        ['subset', 'model', 'model_spec', 'type']).sort_index(level=["subset", "model", "model_spec"], sort_remaining=False).to_string())

    ####################################################
    ## Figure 5: Distributions of cosine similarities ##
    ####################################################
    # Focus on test set of inst2inst, extract all test categories and entities:
    inst2inst_df = pd.read_csv(f'exp2_instantiation/modified_dataset/Pos+Inst2Inst.txt', sep='\t', keep_default_na=False)
    test_df = inst2inst_df.loc[inst2inst_df['InPartition'] == "Test"]
    test_cats = set(test_df.loc[test_df['ExampleType'] == "POS"]['Word2'].unique())
    test_ents = set(list(test_df.loc[test_df['ExampleType'] == "POS"]['Word1'].unique()) + list(test_df.loc[test_df['ExampleType'] == "NEG"]['Word1'].unique()) + list(test_df.loc[test_df['ExampleType'] == "NEG"]['Word2'].unique()))
    all_ents = set(list(inst2inst_df.loc[inst2inst_df['ExampleType'] == "POS"]['Word1'].unique()) + list(inst2inst_df.loc[inst2inst_df['ExampleType'] == "NEG"]['Word1'].unique()) + list(inst2inst_df.loc[inst2inst_df['ExampleType'] == "NEG"]['Word2'].unique()))

    # These entities are used to construct centroids:
    centroid_ents = all_ents - test_ents
    cat_to_centroid_ents = {cat: [ent for ent in cat_to_ent[cat] if ent in centroid_ents] for cat in cat_to_ent}

    # Load embeddings and construct centroids from the training entities
    embs = read_pickled_vectors()
    for key in embs:
        embs[key] = normalize(embs[key])
    embs.update({cat + "!C": centroid([embs[ent] for ent in cat_to_centroid_ents[cat]]) for cat in cat_to_ent})
    embs.update({ent + "!C": embs[ent] for ent in ent_to_cat})

    # Loop through various sub-datasets to compute the cosine similarities:
    test_cat_pairs_df = pd.DataFrame(zip(test_cats, test_cats), columns=["Word1", "Word2"])
    cosines = []
    for kind, df, models in [
        ("ent-ent", test_df.loc[test_df['ExampleType'] == "NEG"], ["none"]),
        ("cat-ent", test_df.loc[test_df['ExampleType'] == "POS"], ["nounbased", "namebased"]),
        ("cat-cat", test_cat_pairs_df, ["nounnamebased"]),
    ]:
        for i, row in df.iterrows():
            word1, word2 = row['Word2'], row['Word1']
            for model in models:
                key1, key2 = word1, word2
                if model == "namebased":
                    key1 += "!C"    # suffix to retrieve centroid representation
                    key2 += "!C"
                elif model == "nounnamebased":
                    key1 += "!C"
                cosine = 1.0 - spatial.distance.cosine(embs[key1], embs[key2])
                cosines.append([kind, model, word1, word2, cosine])
    cosines = pd.DataFrame(cosines, columns=["kind", "model", "word1", "word2", "cosine"])

    # Plot four distributions:
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,6))
    ax = sns.distplot(cosines.loc[(cosines['kind'] == "ent-ent")]["cosine"],
                      label="Inst2Inst: ent/ent (within-domain)")
    sns.distplot(cosines.loc[(cosines['kind'] == "cat-ent") & (cosines['model'] == 'nounbased')]["cosine"],
                 ax=ax,label=f"Pos: ent/NounBased(C)")
    sns.distplot(cosines.loc[(cosines['kind'] == "cat-ent") & (cosines['model'] == 'namebased')]["cosine"],
                 ax=ax, label=f"Pos: ent/NameBased(C)")
    sns.distplot(cosines.loc[(cosines['model'] == 'nounnamebased') & (cosines['kind'] == "cat-cat")]["cosine"],
                 ax=ax,label=f"NounBased(C)/NameBased(C)")
    plt.xlim((-0.2,1.0))
    plt.xlabel("Cosine similarity")
    plt.legend()
    plt.tight_layout()
    path = "plots/distplot.png"
    plt.savefig(path)
    print("Distribution plot saved to", path)
    print("-------------------------------")

    ############################################
    ## Table 8: Exp 2 F1 scores clear/unclear ##
    ############################################
    print("-------------------------------")
    print("Table 8: Exp 2 F1 scores clear/unclear")
    print(results_exp2_master.loc[('Union_inDomain', slice(None), 'nounbased', '2hl'), :].reorder_levels(['type', 'model', 'model_spec', 'subset']).sort_index(level=["type", "model", "model_spec"], sort_remaining=False).to_string())
    print(results_exp2_master.loc[('Union_inDomain', slice(None), 'namebased', '1hl'), :].reorder_levels(['type', 'model', 'model_spec', 'subset']).sort_index(level=["type", "model", "model_spec"], sort_remaining=False).to_string())

    print("-------------------------------")

    #######################################
    ## Table 9: Exp 2 effect of polysemy ##
    #######################################
    print("Table 9: Exp2 F1 correlation with synsets.")
    correlations_exp2 = []
    for partition, path in exp2_results_files[-1:]:   # Only check Combined_tough
        exp2_results = pd.read_csv('exp2_instantiation/' + path)
        exp2_results['category'] = exp2_results.apply(lambda x: x['element1'] if x['type'] in ['Inverse'] else x['element2'] if x['type'] in ["Pos", "NotInst_global", "NotInst_inDomain"] else None, axis=1)
        counts = exp2_results.dropna(subset=['category']).groupby('category')['category'].count()
        print(f'Category counts in {partition}: mean, std, max, min:', counts.mean(), counts.std(), counts.max(), counts.min())
        exp2_results['clear'] = exp2_results['category'].apply(lambda x: polysemy_data.loc[x, 'clear'] if x is not None else None)
        exp2_results['num_synsets'] = exp2_results['category'].apply(lambda x: polysemy_data.at[x, 'num_synsets'] if x is not None else None)

        for subset, df in [
            ('all*', exp2_results.loc[~exp2_results['category'].isna()].copy()),
            ('clear', exp2_results.loc[exp2_results['clear'] == True].copy()),
            ('unclear', exp2_results.loc[exp2_results['clear'] == False].copy()),
        ]:
            for model in ['baseline_freq', 'baseline_pos', 'nounbased_cos', 'nounbased_1hl', 'nounbased_2hl', 'namebased_cos', 'namebased_1hl', 'namebased_2hl']:
                # Logistic:
                df['num_synsets'] = df['num_synsets'] / df['num_synsets'].std()
                df['correct'] = (df['gold'] == df[model]).apply(lambda x: 1 if x else 0)

                fitted = logit('correct ~ num_synsets', df, missing="drop").fit(disp=False)

                summary = list(str(fitted.summary()).split('\n'))
                # Ugly way of extracting info of the model; too lazy to look things up :)
                num_observations = int(summary[2].split('No. Observations:')[1])
                rsquared = float(summary[5].split('Pseudo R-squ.:')[1].strip())
                coef = float(summary[13].split()[1])
                sterr = float(summary[13].split()[2])
                pvalue = fitted.pvalues['num_synsets']

                # Correlation:
                df_synsets = pd.DataFrame([[polysemy_data.loc[x,'num_synsets'], f1_per_category[partition][subset][model][x]] for x in df['category'].unique() if x is not None], columns=['num_synsets', 'f1_score_per_cat'])
                df_synsets_noNA = df_synsets.dropna(subset=['f1_score_per_cat'])
                df_synsets_noNA['num_synsets'] = df_synsets_noNA['num_synsets'] / df_synsets_noNA['num_synsets'].std()
                r, p = spearmanr(df_synsets_noNA['num_synsets'], df_synsets_noNA['f1_score_per_cat'])

                correlations_exp2.append([partition, subset, model.split('_')[0], model.split('_')[1], coef, sterr, pvalue, rsquared, num_observations, r, p, len(df_synsets_noNA)])

    correlations_exp2 = pd.DataFrame(correlations_exp2, columns=['type', 'subset', 'model', 'model_spec', 'logreg-coef', 'std.error', 'logreg-p', 'r-squared', 'num_obs', 'spearman-rho', 'spearman-p', 'num_cats'])
    correlations_exp2.set_index(['type', 'subset', 'model', 'model_spec'], inplace=True)

    # pd.options.display.float_format = '{:,.4f}'.format
    print(correlations_exp2.loc[(['NotInst_inDomain', 'Union_inDomain'], slice(None), 'nounbased', '2hl'), :].reorder_levels(['type', 'model', 'model_spec', 'subset']).sort_index(level=["type", "model", "model_spec"], sort_remaining=False).to_string())
    print(correlations_exp2.loc[(['NotInst_inDomain', 'Union_inDomain'], slice(None), 'namebased', '1hl'), :].reorder_levels(['type', 'model', 'model_spec', 'subset']).sort_index(level=["type", "model", "model_spec"], sort_remaining=False).to_string())
    print("--------------------")


def read_dataset(path):

    df = pd.read_csv(path, delimiter='\t', keep_default_na=False)

    df = df.loc[df['ExampleType'] == "POS"]

    cat_to_ent = {}
    for c in df['Word2'].unique():
        cat_to_ent[c] = df.loc[df['Word2'] == c]['Word1'].unique().tolist()

    # Ugly fix for multi-class categories:
    for cat in list(cat_to_ent.keys()):
        df.loc[df['Word2'] == cat,'OntologicalClass'] = df.loc[df['Word2'] == cat]['OntologicalClass'].mode().at[0]

    df.reset_index(inplace=True)

    ent_to_cat = {ent: cat for cat in cat_to_ent for ent in cat_to_ent[cat]}

    class_to_cat = {}
    for cl in df['OntologicalClass']:
        class_to_cat[cl] = df.loc[df['OntologicalClass'] == cl]['Word2'].unique().tolist()
    cat_to_class = {cat: cl for cl in class_to_cat for cat in class_to_cat[cl]}

    return df, cat_to_ent, ent_to_cat, cat_to_class, class_to_cat


def normalize(vec):
    return vec / np.linalg.norm(vec)


def centroid(vecs, norm=True):
    vec = np.sum(vecs, axis=0) / len(vecs)
    if norm:
        vec = normalize(vec)
    return vec


# helper function: given a dictionary, return a dictionary with the same keys
# that returns ranks, with average ranks for ties
# algorithm: count values and compute ranks based on these frequencies
# (btw, it turns out that this is already implemented in scipy as scipy.stats.rankdata)
def torank_avg(key2value):
    # count frequency of values
    value2freq = defaultdict(int)
    for key, value in key2value.items():
        value2freq[value] += 1
    # now create a sorted list of values
    sorted_values = sorted(set(key2value.values()))
    # now create a dict from values to ranks
    value2rank = {}
    at_rank = 1
    for value in sorted_values:
        #        print ("value "+str(value))
        freq = value2freq[value]
        # compute rank for this value as current rank plus half its frequency minus one
        value2rank[value] = at_rank + (freq - 1) / 2
        at_rank += freq
    # finally, just map all keys onto their ranks
    return {k: value2rank[v] for (k, v) in key2value.items()}


def rank_delta(catps, catp2judgs, catp2preds):
    catp2judgs_subset = {catp: catp2judgs[catp] for catp in catps}
    catp2preds_subset = {catp: catp2preds[catp] for catp in catps}
    ranks = torank_avg(catp2judgs_subset)
    preds = torank_avg(catp2preds_subset)
    delta = {catp: ranks[catp] - preds[catp] for catp in ranks.keys()}
    return delta


def read_pickled_vectors():
    embs = {}
    for i in range(4):
        with open(f'auxiliary_data/noun_and_name_vectors_{i}.pkl', "rb") as picklefile:
            embs.update(pickle.load(picklefile))
    return embs

if __name__ == "__main__":

    main()
