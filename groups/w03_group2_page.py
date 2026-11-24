"""Group 2 - Is the rain effect real?"""
import matplotlib.pyplot as plt
import streamlit as st

from ml_base import load_daily
from w03_group2_rain_test import RainTest

st.title("Is the rain effect real?")
st.caption("Card 2 · Team: Yevhenii, Dzmitry, Darya")

data = load_daily()
model = RainTest(data).fit()

st.header("1️⃣ What we built")
st.markdown("""
`RainTest(data).fit()` compares the average sales of rainy and dry days and
shuffles the rain labels 1,000 times to calculate a two-sided p-value.
""")

st.header("2️⃣ Our answer to the board")
col1, col2 = st.columns(2)
col1.metric("Rainy vs dry sales at 1 mm", f'{model.results_["difference_pct"]:+.1f}%')
col2.metric("Shuffle p-value", f'{model.results_["p_value"]:.3f}')
st.markdown("""
We wouldn't plan campaigns around rain because all four shuffle p-values
were above 0.05. Thursdays averaged £23,657 more than Sundays, and none of
the 1,000 weekday shuffles gave a gap that big.
""")

st.header("🎛️ Try it yourself")
value = st.slider("Rain cutoff (mm)", 1, 10, 1)
model = RainTest(data, threshold=value).fit()
col1, col2, col3 = st.columns(3)
col1.metric("Rainy days", model.results_["rainy_days"])
col2.metric("Rainy vs dry average sales", f'{model.results_["difference_pct"]:+.1f}%')
col3.metric("Shuffle p-value", f'{model.results_["p_value"]:.3f}')
fig = model.plot().figure
st.pyplot(fig)
plt.close(fig)

st.header("3️⃣ Is our number real? + what the AI got wrong")
st.markdown("""
Only seven days reached 10 mm, so a few days can move the average. Pick the
cutoff before looking at the p-values to avoid finding a result by chance.

**AI:** The earlier shuffle test gave p = 0 at a 0 mm cutoff because there
were no dry days. We added a check that both groups have days.
""")
