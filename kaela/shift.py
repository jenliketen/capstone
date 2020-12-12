import numpy as np

def remove_fips_and_shift(df):
    # find fips with missing years between 2007 - 2016
    fipp = list(np.unique(df["fips"]))
    fs = []
    for f in fipp:
        sub = df[df["fips"] == f]
        if np.unique(sub.year).shape[0] != 10:
            fs.append(f)

    # find zips that have all 6 years
    fip_to_keep = list(np.setdiff1d(list(df["fips"].values), fs))

    # only keep data with zips for all 6 years
    sub_fip = df[df["fips"].isin(fip_to_keep)]

    # shift death count and dead variables by 1 year, so outcome is adjusted
    fip_df_new = sub_fip.copy()
    fip_df_new["COPD"] = fip_df_new.groupby(fip_df_new["fips"])["COPD"].shift(-1)
    # fip_df_new["Asthma"] = fip_df_new.groupby(fip_df_new["fips"])["Asthma"].shift(-1)
    fip_df_new.dropna(inplace=True)
    fip_df_new = fip_df_new.reset_index(drop=True)
    
    return fip_df_new