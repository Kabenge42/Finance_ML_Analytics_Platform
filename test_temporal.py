import pandas as pd


def test_temporal_issue():
    df = pd.DataFrame(
        {
            "f_month": pd.Series([1, pd.NA, 3], dtype="Int64"),
            "dates": pd.to_datetime(["2023-01-01", pd.NaT, "2023-03-01"]),
        }
    )

    f_month = df["f_month"]
    dates = df["dates"]

    try:
        # Simulate line 79: result["fiscal_quarter"] = ((f_month - 1) // 3 + 1).fillna(dates.dt.quarter).astype(int)
        fiscal_quarter = ((f_month - 1) // 3 + 1).fillna(dates.dt.quarter)
        print("Fiscal quarter Series before astype(int):")
        print(fiscal_quarter)
        print(f"Dtype: {fiscal_quarter.dtype}")

        # This is where it might fail
        res = fiscal_quarter.astype(int)
        print("Success!")
    except Exception as e:
        print(f"Failed with error: {e}")


if __name__ == "__main__":
    test_temporal_issue()
