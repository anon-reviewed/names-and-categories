import pandas as pd
from scipy.stats.stats import spearmanr
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from collections import defaultdict
from corrstats import dependent_corr
import pickle
import numpy as np
from scipy import spatial
import os
import krippendorff
import warnings


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


    #################################################
    ## Composition of the instantiation dataset >5 ##
    #################################################
    print("Table 2: Composition of the instantiation dataset >5 ")
    print(f"categories:  {len(cat_to_ent)}, instantiations: {sum([len(cat_to_ent[cat]) for cat in cat_to_ent])}, entities: {len(ent_to_cat)}")
    for class_ in class_to_cat:
        print(f"{class_.upper()}:  categories:  {len(class_to_cat[class_])}, instantiations: {len([ent for cat in class_to_cat[class_] for ent in cat_to_ent[cat]])}, entities: {len(set([ent for cat in class_to_cat[class_] for ent in cat_to_ent[cat]]))}")
        print('  ', class_to_cat[class_])
    print("---------------------------------")

    #############################################
    ## Table 3: counts per domain Experiment 1 ##
    #############################################
    print("Table 3. Counts per domain Experiment 1:")
    print(exp1_results.groupby(['cat1_domain', 'within'])['human_relatedness'].count())
    print(exp1_results.groupby(['cat2_domain', 'within'])['human_relatedness'].count())
    print("---------------------------------")

    ############################################
    ## Inter-annotator agreement experiment 1 ##
    ############################################
    print("Inter-annotator agreement analysis experiment 1:\n")
    inter_annotator_agreement()
    print("-------------------------------")

    ##################################
    ### Table 4: Main results exp 1 ##
    ##################################
    print("Table 4 and 6: Main results exp 1.")
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

    ########################################
    #### Significance test experiment 1 ####
    ########################################

    print("Significance matched vs. unclear, experiment 1:")
    bootstrap_exp1(exp1_results)
    print("------------------------")

    ###############################################
    ### Figure 2: Correlation per namebased size ##
    ###############################################
    print("Figure 2: Correlation per namebased size.")
    if input("This takes a long time. Skip? Y/n").lower().startswith("n"):
        N_SAMPLED_NAMEBASED = 50
        np.random.seed(12345)
        auxiliary_path = 'exp1_relatedness/exp1_aggregate_scores_per_namebased_size.csv'
        plot_path = "plots/correlation_per_namebased_size.png"

        # Load model and human scores from auxiliary file
        print("Loading", auxiliary_path)
        with open(auxiliary_path, 'r'):
            all_scores = pd.read_csv(auxiliary_path)

        # Compute correlations many times, with centroids sampled differently
        print("Computing correlation stats for {} different choices of sampled names...".format(N_SAMPLED_NAMEBASED))
        correlations_sampled = []
        for _ in range(N_SAMPLED_NAMEBASED):
            # Sample by taking one from each method (namebased1, namebased2, namebased3, ... predicate) across all category pairs
            sample = all_scores.groupby(['cat1', 'cat2', 'model'], group_keys=False).apply(lambda x: x.sample(1)).sort_index()
            results = correlations_per_namebased_size(sample)
            correlations_sampled.append(results)
        correlations_sampled = pd.concat(correlations_sampled)

        # Plot correlation coefficients
        plot_correlation_per_namebased_size(correlations_sampled, plot_path)

    print("----------------------")

    ##########################################
    ####### SCATTERPLOTS APPENDIX EXP 1 ######
    ##########################################

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
                "UndefinedMetricWarning"
                model_prefix, model_suffix = model.split('_')
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
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
    ## Table 5: Exp 2 F1 scores all models/datasets ##
    ##################################################
    # pd.options.display.float_format = '{:,.2f}'.format

    print("Table 5: Exp 2 F1 scores all models/datasets")
    print(results_exp2_master.loc[(slice(None), 'all', slice(None), slice(None)), :].reorder_levels(
        ['subset', 'model', 'model_spec', 'type']).sort_index(level=["subset", "model", "model_spec"], sort_remaining=False).to_string())
    print("-------------------------------")

    ##############################################
    ## Table 7: Exp 2 F1 scores matched/unclear ##
    ##############################################
    print("Table 7: Exp 2 F1 scores clear/unclear")
    print(results_exp2_master.loc[('Union_inDomain', slice(None), 'nounbased', '2hl'), :].reorder_levels(['type', 'model', 'model_spec', 'subset']).sort_index(level=["type", "model", "model_spec"], sort_remaining=False).to_string())
    print(results_exp2_master.loc[('Union_inDomain', slice(None), 'namebased', '1hl'), :].reorder_levels(['type', 'model', 'model_spec', 'subset']).sort_index(level=["type", "model", "model_spec"], sort_remaining=False).to_string())

    print("-------------------------------")

    #########################################
    #### Significance tests experiment 2 ####
    #########################################

    print("Significance matched vs. unclear, experiment 2:")
    exp2_results_files = [
        # ('Union_global', 'exp2_predictions_Union_global.csv'),
        ('Union_inDomain', 'exp2_predictions_Union_inDomain.csv'),
    ]
    for partition, path in exp2_results_files:
        print('  Analysis for subset:', partition)
        exp2_results = pd.read_csv('exp2_instantiation/' + path)
        exp2_results['category'] = exp2_results.apply(lambda x: x['element1'] if x['type'] in ['Inverse'] else x['element2'] if x['type'] in ["Pos", "NotInst_global", "NotInst_inDomain"] else None, axis=1)
        exp2_results['clear'] = exp2_results['category'].apply( lambda x: polysemy_data.loc[x, 'clear'] if x is not None else None)
        bootstrap_exp2(exp2_results)

    print("-------------------------------")


    ###############################################################
    ## Figure 4 (Appendix): Distributions of cosine similarities ##
    ###############################################################
    distplots(cat_to_ent, ent_to_cat)


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


def correlations_per_namebased_size(scores_df):
    """
    Compute correlation for all models, given one way of sampling namebased representations.
    :param scores_df: Dataframe as read from data/exp1_aggregate_scores_per_namebased_size.csv.
    :return: dataframe with correlation coefficients per model
    """

    # Ugly maneuver to get method as columns whilst maintaining other info:
    means = scores_df[['cat1', 'cat2', 'model', 'model_cos']].groupby(['cat1', 'cat2', 'model']).mean().unstack()['model_cos'].reset_index()
    other_info = scores_df[['cat1', 'cat2', 'model', 'n_ent1', 'n_ent2', 'n_ent_min', 'human_relatedness']].groupby(['cat1', 'cat2']).agg(lambda x: x.value_counts().index[0])
    scores_df = pd.merge(means, other_info, how='left', on=['cat1', 'cat2'])

    # Compute Spearman correlations for each type of representation (namebased1-5max, nounbased)
    correlations = {}      # between model and human.

    # Loop over all methods (variable n is either an integer or "max"; and label is "namebased" or "nounbased".)
    for n, label in zip(list(range(1, 6)) + ["max"] + list(range(1, 6)) + ["max"],
                        ["namebased"] * 6 + ["nounbased"] * 6):
        # First restrict the dataframe to the relevant points, and change the labels; sorry, bit ugly!
        if isinstance(n, int):
            df = scores_df.loc[scores_df["n_ent_min"] >= n]
            if label == "namebased":
                oldlabel = label + str(n)
                newlabel = oldlabel
            if label == "nounbased":
                oldlabel = label
                newlabel = label + str(n)
        else:
            df = scores_df
            oldlabel = label if label == "nounbased" else (label + "max")
            newlabel = label + "max"

        # Now compute correlations (model~human)
        correlations[newlabel] = {}
        r, p = spearmanr(df[oldlabel], df['human_relatedness'])   # TODO avoid reliance on oldlabel
        correlations[newlabel][("spearman", "all")] = r
        correlations[newlabel][("spearman_p", "all")] = p

    # Some Pandas magic to get a useful dataframe
    results = pd.DataFrame(correlations)
    results = results.transpose()
    results = results.stack(1).reset_index()
    results["model"] = results["level_0"].apply(lambda x: "namebased" if x.startswith("namebased") else "nounbased")
    results[("size")] = results["level_0"].apply(lambda x: x[len("namebased"):] if x.startswith("namebased") else x[len("nounbased"):])
    del results['level_0']
    results = results.rename(columns={"level_1": "class", "level_2": "measure"})

    results.loc[results['size'] == -1, 'size'] = 'max'
    results['size'].apply(str)

    return results


def plot_correlation_per_namebased_size(correlations, plot_path):
    """
    Creates figure 4 in the paper, a plot of correlation coefficient per number of names used for the namebased representation.
    :param correlations: dataframe as created by correlations_per_namebased_size()
    :param plot_path: where the plot is saved
    """
    correlations.replace(to_replace={'size': {'max': 'all'}}, inplace=True)

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 16})

    ax = sns.barplot(x="size", y="spearman", hue="model", data=correlations.loc[correlations['model'] == 'namebased'], ci="sd")
    ax.axhline(0.560372, ls='--', color="black")
    ax.text(-.4, 0.58, "NounBased")

    plt.ylim(0.17, .83)     # For consistency (though values can be sub-zero...)
    plt.xlabel("# of names per category for NameBased model")
    plt.ylabel("Spearman's rho")

    plt.tight_layout()
    plt.savefig(plot_path)
    print("Plot saved as", plot_path)
    plt.close()


def inter_annotator_agreement():
    judgments = pd.read_csv('exp1_relatedness/exp1_agreement_judgments_anon.csv').reset_index()
    judgments = judgments.loc[
        ~judgments['mturk_id'].isin(['person_0', 'person_4', 'person_5'])]  # remove unreliable annotators
    judgments_filler = judgments.loc[~judgments['target']]
    judgments_target = judgments.loc[judgments['target']]

    mean_choices = judgments_target.groupby(['cat1', 'cat2', 'cat3', 'cat4']).agg({'choice': 'mean'})
    print('Most divided:\n', mean_choices.loc[mean_choices['choice'].isin([0.4, 0.6])])

    target_per_annotator = judgments_target.sort_values(['cat1', 'cat2', 'cat3', 'cat4']).groupby('mturk_id').agg(
        {'choice': lambda x: x.tolist()})
    filler_per_annotator = judgments_filler.sort_values(['cat1', 'cat2', 'cat3', 'cat4']).groupby('mturk_id').agg(
        {'choice': lambda x: x.tolist()})
    target_per_annotator_list = target_per_annotator['choice'].tolist()
    filler_per_annotator_list = filler_per_annotator['choice'].tolist()
    filler_per_annotator_list = [ans[:5] + ans[6:] for ans in
                                 filler_per_annotator_list]  # Remove filler 5 (lack of agreement in original data)
    all_per_annotator_list = [targets + fillers for targets, fillers in
                              zip(target_per_annotator_list, filler_per_annotator_list)]

    # Krippendorff fails since all targets are encoded as 0, so invert half of the judgments:
    for rater in range(len(filler_per_annotator_list)):
        for filler in range(len(filler_per_annotator_list[rater])):
            if filler % 2 == 1:
                filler_per_annotator_list[rater][filler] = 1 - filler_per_annotator_list[rater][filler]

    print("\nKrippendorff:")
    print('all:', krippendorff.alpha(all_per_annotator_list))
    print('targets:', krippendorff.alpha(target_per_annotator_list))
    print('fillers:', krippendorff.alpha(filler_per_annotator_list))


def bootstrap_exp2(exp2_results):

    micro_contrasts_nounbased = []
    micro_contrasts_namebased = []
    macro_contrasts_nounbased = []
    macro_contrasts_namebased = []

    micro_difs = []
    macro_difs = []

    dfs = {'all': exp2_results,
           'all*': exp2_results.loc[~exp2_results['category'].isna()].copy(),
           'clear': exp2_results.loc[exp2_results['clear'] == True],
           'unclear': exp2_results.loc[exp2_results['clear'] == False]}

    for _ in range(1000):

        clear = dfs['clear'].sample(frac=1, replace=True)
        unclear = dfs['unclear'].sample(frac=1, replace=True)

        micro_f1_namebased_clear = f1_score(clear['gold'], clear['namebased_1hl'], pos_label="Yes")
        micro_f1_namebased_unclear = f1_score(unclear['gold'], unclear['namebased_1hl'], pos_label="Yes")
        micro_f1_nounbased_clear = f1_score(clear['gold'], clear['nounbased_2hl'], pos_label="Yes")
        micro_f1_nounbased_unclear = f1_score(unclear['gold'], unclear['nounbased_2hl'], pos_label="Yes")

        clear = clear.loc[~clear['category'].isna()]
        unclear = unclear.loc[~unclear['category'].isna()]

        clear_cat_nunique = clear.groupby('category')['gold'].apply(lambda x: len(set(x)))
        unclear_cat_nunique = unclear.groupby('category')['gold'].apply(lambda x: len(set(x)))

        clear = clear.loc[clear['category'].apply(lambda x: clear_cat_nunique[x] == 2)]
        unclear = unclear.loc[unclear['category'].apply(lambda x: unclear_cat_nunique[x] == 2)]

        macro_f1_namebased_clear = clear.groupby('category').apply(lambda x: f1_score(x['gold'], x['namebased_1hl'], pos_label="Yes")).mean()
        macro_f1_nounbased_clear = clear.groupby('category').apply(lambda x: f1_score(x['gold'], x['nounbased_2hl'], pos_label="Yes")).mean()
        macro_f1_namebased_unclear = unclear.groupby('category').apply(lambda x: f1_score(x['gold'], x['namebased_1hl'], pos_label="Yes")).mean()
        macro_f1_nounbased_unclear = unclear.groupby('category').apply(lambda x: f1_score(x['gold'], x['nounbased_2hl'], pos_label="Yes")).mean()

        # micro_f1_namebased = 1/(1/micro_f1_namebased_clear - 1/micro_f1_namebased_unclear) # / micro_f1_namebased_clear
        # micro_f1_nounbased = 1/(1/micro_f1_nounbased_clear - 1/micro_f1_nounbased_unclear) # / micro_f1_nounbased_clear
        # macro_f1_namebased = 1/(1/macro_f1_namebased_clear - 1/macro_f1_namebased_unclear) # / macro_f1_namebased_clear
        # macro_f1_nounbased = 1/(1/macro_f1_nounbased_clear - 1/macro_f1_nounbased_unclear) # / macro_f1_nounbased_clear
        #
        # micro_difs.append(1/(1/micro_f1_nounbased - 1/micro_f1_namebased))
        # macro_difs.append(1/(1/macro_f1_nounbased - 1/macro_f1_namebased))

        micro_f1_namebased = (micro_f1_namebased_clear - micro_f1_namebased_unclear) # / micro_f1_namebased_clear
        micro_f1_nounbased = (micro_f1_nounbased_clear - micro_f1_nounbased_unclear) # / micro_f1_nounbased_clear
        macro_f1_namebased = (macro_f1_namebased_clear - macro_f1_namebased_unclear) # / macro_f1_namebased_clear
        macro_f1_nounbased = (macro_f1_nounbased_clear - macro_f1_nounbased_unclear) # / macro_f1_nounbased_clear

        macro_contrasts_namebased.append(macro_f1_namebased)
        micro_contrasts_namebased.append(micro_f1_namebased)
        macro_contrasts_nounbased.append(macro_f1_nounbased)
        micro_contrasts_nounbased.append(micro_f1_nounbased)

        micro_difs.append(micro_f1_nounbased - micro_f1_namebased)
        macro_difs.append(macro_f1_nounbased - macro_f1_namebased)

    print("   - Macro contrasts nounbased:")
    percentiles(macro_contrasts_nounbased)
    print("   - Macro contrasts namebased:")
    percentiles(macro_contrasts_namebased)

    print("   - Micro contrasts nounbased:")
    percentiles(micro_contrasts_nounbased)
    print("   - Micro contrasts namebased:")
    percentiles(micro_contrasts_namebased)

    print("   - Micro difs:")
    percentiles(micro_difs)

    print("   - Macro difs:")
    percentiles(macro_difs)


def bootstrap_exp1(exp1_results):

    difs = []
    difs_nounbased = []
    difs_namebased = []

    for _ in range(1000):
        match = exp1_results.loc[exp1_results['clear']]
        unclear = exp1_results.loc[~exp1_results['clear']]

        match_sample = match.sample(n=len(match), replace=True)
        unclear_sample = unclear.sample(n=len(unclear), replace=True)

        match_nounbased, _ = spearmanr(match_sample['nounbased_cos'], match_sample['human_relatedness'])
        match_namebased, _ = spearmanr(match_sample['namebased_cos'], match_sample['human_relatedness'])
        unclear_nounbased, _ = spearmanr(unclear_sample['nounbased_cos'], unclear_sample['human_relatedness'])
        unclear_namebased, _ = spearmanr(unclear_sample['namebased_cos'], unclear_sample['human_relatedness'])

        dif_nounbased = (match_nounbased - unclear_nounbased)
        dif_namebased = (match_namebased - unclear_namebased)

        difs_nounbased.append(dif_nounbased)
        difs_namebased.append(dif_namebased)
        difs.append(dif_nounbased - dif_namebased)

    print("- Overall:")
    percentiles(difs)

    print("- Nounbased")
    percentiles(difs_nounbased)

    print("- Namebased")
    percentiles(difs_namebased)


def percentiles(measurements):
    ms = sorted(measurements)
    # print(".1st perc. " + str(ms[round(len(ms) / 1000)]))
    # print(" 1st perc. " + str(ms[round(len(ms) / 100)]))
    # print(" 5th perc. " + str(ms[round(len(ms) / 20)]))
    # print("50th perc. " + str(ms[round(len(ms) / 2)]))
    # print("95th perc. " + str(ms[round(len(ms) - len(ms) / 20)]))
    # print("99th perc. " + str(ms[round(len(ms) - len(ms) / 100)]))
    # print("99.9 perc. " + str(ms[round(len(ms) - len(ms) / 1000)]))
    print("  mean: " + str(np.mean(ms)))
    for i, m in enumerate(ms):
        if m > 0:
            print("  p-value: " + str(i/len(ms)))
            break


def distplots(cat_to_ent, ent_to_cat):
    # Focus on test set of inst2inst, extract all test categories and entities:
    inst2inst_df = pd.read_csv(f'exp2_instantiation/modified_dataset/Pos+Inst2Inst.txt', sep='\t',
                               keep_default_na=False)
    test_df = inst2inst_df.loc[inst2inst_df['InPartition'] == "Test"]
    test_cats = set(test_df.loc[test_df['ExampleType'] == "POS"]['Word2'].unique())
    test_ents = set(list(test_df.loc[test_df['ExampleType'] == "POS"]['Word1'].unique()) + list(
        test_df.loc[test_df['ExampleType'] == "NEG"]['Word1'].unique()) + list(
        test_df.loc[test_df['ExampleType'] == "NEG"]['Word2'].unique()))
    all_ents = set(list(inst2inst_df.loc[inst2inst_df['ExampleType'] == "POS"]['Word1'].unique()) + list(
        inst2inst_df.loc[inst2inst_df['ExampleType'] == "NEG"]['Word1'].unique()) + list(
        inst2inst_df.loc[inst2inst_df['ExampleType'] == "NEG"]['Word2'].unique()))

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
                    key1 += "!C"  # suffix to retrieve centroid representation
                    key2 += "!C"
                elif model == "nounnamebased":
                    key1 += "!C"
                cosine = 1.0 - spatial.distance.cosine(embs[key1], embs[key2])
                cosines.append([kind, model, word1, word2, cosine])
    cosines = pd.DataFrame(cosines, columns=["kind", "model", "word1", "word2", "cosine"])

    # Plot four distributions:
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(cosines.loc[(cosines['kind'] == "ent-ent")]["cosine"],
                     label="Inst2Inst: ent/ent (within-domain)", kde=True, stat="density", linewidth=0)
    sns.histplot(cosines.loc[(cosines['kind'] == "cat-ent") & (cosines['model'] == 'nounbased')]["cosine"],
                ax=ax, label=f"Pos: ent/NounBased(C)", kde=True, stat="density", linewidth=0, color="orange")
    sns.histplot(cosines.loc[(cosines['kind'] == "cat-ent") & (cosines['model'] == 'namebased')]["cosine"],
                ax=ax, label=f"Pos: ent/NameBased(C)", kde=True, stat="density", linewidth=0, color="green")
    sns.histplot(cosines.loc[(cosines['model'] == 'nounnamebased') & (cosines['kind'] == "cat-cat")]["cosine"],
                ax=ax, label=f"NounBased(C)/NameBased(C)", kde=True, stat="density", linewidth=0, color="red")
    plt.xlim((-0.2, 1.0))
    plt.xlabel("Cosine similarity")
    plt.legend()
    plt.tight_layout()
    path = "plots/distplot.png"
    plt.savefig(path)
    print("Distribution plot saved to", path)
    print("-------------------------------")


if __name__ == "__main__":

    main()
