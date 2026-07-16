'''
Mini-Project: World Happiness Pipeline
'''

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from prefect import task, flow, get_run_logger


# ==========================================
#  Task 1: Load Multiple Years of Data
# ==========================================

# INSTRUCTION: (@task): Load data from all ten yearly CSV files into a single DataFrame
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent/ 'assignments_01' / 'resources' / 'happiness_project'

OUTPUT_DIR ='outputs'

@task(retries=3, retry_delay_seconds=2)
def load_data(data_dir):
    logger = get_run_logger()
    logger.info(f"Scanning for data files in: {data_dir}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    #each year dataframe
    all_data = []

    ## INSTRUCTION: Your implementation should not duplicate code for each year -- iterate over a list of file paths and load them in a loop.
    for year in range(2015,2025):
        filepath = os.path.join(data_dir,f"world_happiness_{year}.csv")

        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep=";", decimal=",")

            COLUMN_MAP = {
                "ladder_score": "happiness_score",
                "Happiness score": 'happiness_score',
            }
            
            #Standardize columns and I found 2024 happiness score is called ladder score
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            df = df.rename(columns= COLUMN_MAP)

            ## INSTRUCTION: each row needs to know which year it came from.
            #added year into columns
            df["year"] = year
            
            all_data.append(df)
            logger.info(f"Loaded {year} data: {len(df)} rows.")
        else:
            logger.warning(f"File not found for year {year}: {filepath}")

    #MERGE: pd.concat
    merged_df = pd.concat(all_data, ignore_index=True)

    ## INSTRUCTION: save the combined dataset to: 'assignments_01/outputs/merged_happiness.csv'
    output_path = os.path.join(OUTPUT_DIR,'merged_happiness.csv')
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Merged data saved to {output_path} with {len(merged_df)} total rows.")
    
    return merged_df


# ==========================================
# Task 2: Descriptive Statistics
# ==========================================

@task
def descriptive_statistics(df):
    logger = get_run_logger()

    #Compute and log overall descriptive statistics for happiness_score: mean, median, and standard deviation.

    #----OVERALL----#
    target_col = "happiness_score"
    
    mean = df[target_col].mean()
    median = df[target_col].median()
    std = df[target_col].std()

    logger.info(f"overall stats:\n")
    logger.info(f"Overall {target_col.capitalize} Stats: ")
    logger.info(f"Mean: {mean}")
    logger.info(f"Median: {median}")
    logger.info(f"Standard Deviation: {std}")

    # Then compute and log the mean happiness score grouped by year and by region

    #----YEAR-----#
    yearly_mean = df.groupby("year")[target_col].mean()
    logger.info(f"\nYearly mean stats: \n{yearly_mean}")

    #----REGION----#
    regional_mean = df.groupby("regional_indicator")[target_col].mean().sort_values(ascending=False)
    logger.info(f"\nMean by region stats:\n{regional_mean}")

# ==========================================
# Task 3: Visual Exploration
# ==========================================
@task
def visual_exploration(df):
    logger = get_run_logger()

    #Histogram: A histogram of all happiness scores across all years. Save as happiness_histogram.png.

    plt.figure()
    sns.histplot(df['happiness_score'])
    plt.title("Distribution of Happiness Scores (All Years)")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "happiness_histogram.png"))
    plt.close()
    logger.info("Saved happiness_histogram.png")

    #Boxplot: A boxplot comparing happiness score distributions across years (one box per year). Save as happiness_by_year.png.
    plt.figure()
    sns.boxplot(data=df, x="year", y="happiness_score")
    plt.title("Happiness Score Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.savefig(os.path.join(OUTPUT_DIR, "happiness_by_year.png"))
    plt.close()
    logger.info("Saved happiness_by_year.png")
    
    #Scatterplot: A scatter plot showing the relationship between GDP per capita and happiness score. Save as gdp_vs_happiness.png.
    plt.figure()
    sns.scatterplot(data=df, x='gdp_per_capita',y='happiness_score')
    plt.title("GDP vs. Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.savefig(os.path.join(OUTPUT_DIR, "gdp_vs_happiness.png"))
    plt.close()
    logger.info("Saved gdp_vs_happiness.png")

    #HEATMAP: A correlation heatmap (using sns.heatmap() with annot=True) showing the Pearson correlations between all numeric columns. Save as correlation_heatmap.png.
    #numeric dtypes:
    numeric_df = df.select_dtypes(include='number')
    plt.figure()
    sns.heatmap(data=numeric_df.corr(),annot=True)
    plt.title("Correlation Heatmap of Numeric Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()
    logger.info("Saved correlation_heatmap.png")

# ==========================================
# Task 4: Hypothesis Testing
# ==========================================
@task
def hypothesis_testing(df):
    logger = get_run_logger()

    #TEST 1

    # Test this directly: run an independent samples t-test comparing happiness scores from 2019 to 2020.
    # t-statistic, p-value, the mean happiness for each group, and a plain-language interpretation of the result at alpha = 0.05.

    #df by year
    happy_scores_2019 = df[df['year']== 2019]['happiness_score'].dropna()
    happy_scores_2020 = df[df['year']== 2020]['happiness_score'].dropna()

    #means
    mean_2019 = happy_scores_2019.mean()
    mean_2020 = happy_scores_2020.mean()

    #tstat,pval
    t_stat, p_val = stats.ttest_ind(happy_scores_2019, happy_scores_2020)

    logger.info(f"2019 vs 2020 Comparison:")
    logger.info(f"Mean 2019: {mean_2019:.3f} | Mean 2020: {mean_2020:.3f}")
    logger.info(f"T-statistic: {t_stat:.4f} | P-value: {p_val:.4f}")

    if p_val <0.05:
        logger.info('There is a statistical difference in happiness between 2019 and 2020. This means the pandemic did affect the happiness of people between the years.')
    else:
        logger.info('There is NO statistical difference in happiness between 2019 and 2020. Meaning, the average happiness stayed relatively the same even with the pandemic.')
    

    #TEST 2
    # Add a second test of your choice (for example, comparing two specific regions that you expect to differ based on the descriptive statistics you computed earlier).
    # regional_indicator

    we_scores = df[df['regional_indicator']=='Western Europe']['happiness_score']
    na_scores = df[df['regional_indicator']=='North America and ANZ']['happiness_score']

    #means
    we_mean= we_scores.mean()
    na_mean=na_scores.mean()

    #tstat,pval
    t_stat_2,p_val_2 = stats.ttest_ind(we_scores,na_scores)

    logger.info(f"Western Europe vs North America and ANZ Comparison:")
    logger.info(f"Mean Western Europe: {we_mean:.3f} | Mean North America and ANZ: {na_mean:.3f}")
    logger.info(f"T-statistic: {t_stat_2:.4f} | P-value: {p_val_2:.4f}")

    if p_val_2 <0.05:
        logger.info('There is a statistical difference in happiness between Western Europe and North America. This means there is a big difference in happiness betweeen ech region.')
    else:
        logger.info('There is NO statistical difference in happiness between Western Europe and North America. Meaning, the average happiness stayed relatively the same even in different regions.')

    return {"mean_2019": mean_2019, "mean_2020": mean_2020, "p_val_1": p_val}

# ==========================================
# Task 5: Correlation and Multiple Comparisons
# ==========================================
@task
def correlation_analysis(df):
    logger = get_run_logger()

    #For each numeric explanatory variable, compute the Pearson correlation with happiness score using scipy.stats.pearsonr and log the coefficient and p-value.
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    #I took out these because they dont relate to the score.
    numeric_cols = [c for c in numeric_cols if c not in ['happiness_score', "year", 'ranking']]
    
    num_tests = len(numeric_cols)
    # adjusted_alpha = 0.05 / number_of_tests 
    # Log which correlations are significant at the original alpha = 0.05, and which remain significant after applying the correction. 
    # You may find that some results that looked significant at first don't hold up under the stricter threshold -- that's a useful finding in itself.
    
    original_alpha = 0.05
    adjusted_alpha = original_alpha / num_tests
    
    logger.info(f"Running {num_tests} correlation tests. Adjusted Alpha: {adjusted_alpha:.5f}")
    
    strongest_sig_var = None
    max_corr = 0
    strongest_overall = None
    max_abs_r = 0
    
    for col in numeric_cols:
        valid_data = df[[col, 'happiness_score']].dropna()
        if len(valid_data) > 2:
            r, p = stats.pearsonr(valid_data[col], valid_data['happiness_score'])

            # Track strongest overall (absolute value)
            if abs(r) > abs(max_abs_r):
                max_abs_r = abs(r)
                strongest_overall = col

            # Track strongest significant (Bonferroni)
            if p < adjusted_alpha and abs(r) > abs(max_corr):
                max_corr = abs(r)
                strongest_sig_var = col

            sig_original = "YES" if p < original_alpha else "NO"
            sig_adjusted = "YES" if p < adjusted_alpha else "NO"
            
            logger.info(f"Var: {col} | r: {r:+.4f} | p: {p:.5e} | Sig (0.05): {sig_original} | Sig (Adj): {sig_adjusted}")
                
    return strongest_sig_var, strongest_overall


# ==========================================
# Task 6: Summary Report
# ==========================================
@task
def summary_report(df, hypothesis_results, strongest_var,strongest_all):
    logger = get_run_logger()
    
    logger.info("==========================================")
    logger.info("          FINAL PIPELINE REPORT           ")
    logger.info("==========================================")
    
    # Total number of countries and years in the merged dataset.
    total_rows = len(df)
    total_years = df["year"].nunique()
    total_countries = df['country'].nunique()

    logger.info(f"- Dataset: {total_countries} unique countries across {total_years} years ({total_rows} total records).")
    
    # The top 3 and bottom 3 regions by mean happiness score.
    regional_mean = df.groupby('regional_indicator')['happiness_score'].mean().sort_values(ascending=False)
    logger.info(f"- Top 3 happiest regions:")
    for i in range(3):
        logger.info(f"  {i+1}. {regional_mean.index[i]} (Mean: {regional_mean.iloc[i]:.2f})")
        
    logger.info(f"- Bottom 3 happiest regions:")
    for i in range(1, 4):
        idx = -i
        logger.info(f"  {i}. {regional_mean.index[idx]} (Mean: {regional_mean.iloc[idx]:.2f})")
        
    # The result of the pre/post-2020 t-test in plain language.
    pval = hypothesis_results["p_val_1"]
    if pval < 0.05:
        logger.info(f"- Statistically significant shift during the pandemic (2019 vs 2020) (p-value: {pval:.4e}).")
    else:
        logger.info(f"- NO statistically significant shift during the pandemic (2019 vs 2020) (p-value: {pval:.4e}).")
        
    # 4. The variable most strongly correlated with happiness score (after Bonferroni correction).
    logger.info(f"- Strongest correlation (Bonferroni significant): {strongest_var if strongest_var else 'None'}")
    logger.info(f"- Strongest correlation (Overall strongest absolute value): {strongest_all}")

##### FLOW CODE #####
# 2. FLow
@flow(name="World Happiness Pipeline")
def happiness_pipeline():
    logger = get_run_logger()
    logger.info("Starting the pipeline...")
    
    # Task 1: Load Multiple Years of Data
    df = load_data(DATA_DIR)

    # Task 2: Descriptive Statistics
    descriptive_statistics(df)

    # Task 3: Visual Exploration
    visual_exploration(df)

    # Task 4: Hypothesis Testing
    hypothesis_results = hypothesis_testing(df)

    # Task 5: Correlation and Multiple Comparisons
    strongest_sig, strongest_all = correlation_analysis(df)

    # Task 6: Summary Report
    summary_report(df, hypothesis_results, strongest_sig, strongest_all)

# 3. Main
if __name__ == "__main__":
    # When you run this file in the terminal, this block catches it and fires the flow.
    happiness_pipeline()