class Annotator:
    def __init__(self) -> None:
        self.nu_desc_threshold = 1

    def annotate_binary(self, df):
        '''
        annotating the target variable based on a NuDesc threshold
                Parameters: 
                        df (df): the master table
                Returns:
                        df (df): master table with annotated target variable
        '''

        # using the set nu threshold for binary annotation
        df = df.assign(
            c_target=[
                1 if x >= self.nu_desc_threshold else 0 for x in df["c_value"]]
        )

        # positive if at least one is positive
        df_pos = df[df.c_target == 1]
        # negative if all assessments are negative
        df_neg = df[(df.c_target == 0) & (df.c_value != 1)]
        df_neg = df_neg[~df_neg.c_link_id.isin(df_pos.c_link_id)]

        # print group sizes and return joined cohort
        print(
            "# positives NuDesc threshold at", self.nu_desc_threshold, ":", len(
                df_pos)
        )
        print(
            "# negatives NuDesc threshold at", self.nu_desc_threshold, ":", len(
                df_neg)
        )

        return df_neg.append(df_pos).rename(columns={"c_value": "c_nudesc"})
