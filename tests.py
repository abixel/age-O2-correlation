from scipy.stats import spearmanr,mannwhitneyu,ttest_ind

test_names = {'Spearman': "Spearman",
              'MannWhitney': "Mann-Whitney",
              'Welch': "Welch",
              'Student': "Student"}

def Spearman(ages,O2,p_only=True):
    rho,p = spearmanr(ages,O2.astype(int))
    return p if p_only else (p,rho)
    
def MannWhitney(ages,O2,p_only=True):
    U,p = mannwhitneyu(ages[O2],ages[~O2],alternative='greater')
    return p if p_only else (p,U)
        
def Welch(ages,O2,p_only=True):
    t,p = ttest_ind(ages[O2],ages[~O2],equal_var=False)
    return p if p_only else (p,t)

def Student(ages,O2,p_only=True):
    t,p = ttest_ind(ages[O2],ages[~O2])
    return p if p_only else (p,t)    



