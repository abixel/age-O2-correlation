from scipy.stats import spearmanr,mannwhitneyu,ttest_ind

test_names = {'Spearman': "Spearman",
              'MannWhitney': "Mann-Whitney",
              'Welch': "Welch",
              'Student': "Student"}

def Spearman(ages,O2,p_only=True,onesided=True):
    rho,p = spearmanr(ages,O2.astype(int))
    p = p/2. if onesided else p
    return p if p_only else (p,rho)
    
def MannWhitney(ages,O2,p_only=True,onesided=True):
    U,p = mannwhitneyu(ages[O2],ages[~O2],alternative='greater' if onesided else 'two-sided')
    return p if p_only else (p,U)
        
def Welch(ages,O2,p_only=True,onesided=True):
    t,p = ttest_ind(ages[O2],ages[~O2],equal_var=False)
    p = p/2. if onesided else p
    return p if p_only else (p,t)

def Student(ages,O2,p_only=True,onesided=True):
    t,p = ttest_ind(ages[O2],ages[~O2])
    p = p/2. if onesided else p
    return p if p_only else (p,t)    



