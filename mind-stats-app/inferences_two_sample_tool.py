                themed_box("**Step 2: Test Statistic**")
                st.latex(r"F = \frac{s_1^2}{s_2^2}")
                st.latex(fr"s_1 = {s1:.{dec}f}, \quad s_2 = {s2:.{dec}f}")
                st.latex(fr"F = {F:.{dec}f}")

                themed_box("**Step 3: Classical Method**")
                p_val, reject, crit = f_tail_metrics(F, df1, df2, alpha, tails)

                if tails == "two":
                    crit_str = f"{crit[0]:.{dec}f}, {crit[1]:.{dec}f}"
                    st.latex(fr"\text{{Critical values}} = {crit[0]:.{dec}f},\ {crit[1]:.{dec}f}")
                    st.markdown(
                        f"Decision rule: Reject H₀ if F < {crit[0]:.{dec}f} or F > {crit[1]:.{dec}f}."
                    )
                    reject_classical = (F < crit[0]) or (F > crit[1])
                    crit_vals = crit
                else:
                    crit_str = f"{crit:.{dec}f}"
                    st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                    if tails == "left":
                        st.markdown(f"Decision rule: Reject H₀ if F < {crit:.{dec}f}.")
                        reject_classical = F < crit
                    else:
                        st.markdown(f"Decision rule: Reject H₀ if F > {crit:.{dec}f}.")
                        reject_classical = F > crit
                    crit_vals = crit

                st.markdown(f"Observed test statistic: F = **{F:.{dec}f}**")
                st.markdown(f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**")

                if show_picture:
                    plot_f_rejection(F, df1, df2, alpha, tails, crit_vals)

                themed_box("**Step 4: P-value Approach**")
                st.markdown(f"P-value = **{p_val:.{dec}f}**")
                st.markdown(f"α = **{alpha:.{dec}f}**")
                st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")

                themed_box("**Step 5: Conclusion**")
                st.write(
                    conclusion_text(
                        reject,
                        "There is sufficient evidence to support the claim in H₁ about the population variances.",
                        "There is not sufficient evidence to support the claim in H₁ about the population variances."
                    )
                )

                if show_ci:
                    ci_message()
                    display_ci("variance", F, None, (df1, df2), ci_conf_level, dec)

        else:
            n1 = st.number_input("n₁:", min_value=2, step=1)
            s1 = st.number_input("s₁:", min_value=0.0, value=1.0)
            n2 = st.number_input("n₂:", min_value=2, step=1)
            s2 = st.number_input("s₂:", min_value=0.0, value=1.0)

            if st.button("▶️ Run Test"):
                if s1 == 0 or s2 == 0:
                    st.error("Both sample standard deviations must be greater than 0.")
                    return

                df1, df2 = n1 - 1, n2 - 1
                F = (s1**2) / (s2**2)
                h0, h1 = get_f_hypotheses(tails)

                st.markdown("### 📘 Step-by-Step Solution")

                themed_box("**Step 1: Hypotheses**")
                st.latex(h0)
                st.latex(h1)

                themed_box("**Step 2: Test Statistic**")
                st.latex(r"F = \frac{s_1^2}{s_2^2}")
                st.latex(fr"s_1 = {s1:.{dec}f}, \quad s_2 = {s2:.{dec}f}")
                st.latex(fr"F = {F:.{dec}f}")

                themed_box("**Step 3: Classical Method**")
                p_val, reject, crit = f_tail_metrics(F, df1, df2, alpha, tails)

                if tails == "two":
                    crit_str = f"{crit[0]:.{dec}f}, {crit[1]:.{dec}f}"
                    st.latex(fr"\text{{Critical values}} = {crit[0]:.{dec}f},\ {crit[1]:.{dec}f}")
                    st.markdown(
                        f"Decision rule: Reject H₀ if F < {crit[0]:.{dec}f} or F > {crit[1]:.{dec}f}."
                    )
                    reject_classical = (F < crit[0]) or (F > crit[1])
                    crit_vals = crit
                else:
                    crit_str = f"{crit:.{dec}f}"
                    st.latex(fr"\text{{Critical value}} = {crit:.{dec}f}")
                    if tails == "left":
                        st.markdown(f"Decision rule: Reject H₀ if F < {crit:.{dec}f}.")
                        reject_classical = F < crit
                    else:
                        st.markdown(f"Decision rule: Reject H₀ if F > {crit:.{dec}f}.")
                        reject_classical = F > crit
                    crit_vals = crit

                st.markdown(f"Observed test statistic: F = **{F:.{dec}f}**")
                st.markdown(f"Classical method decision: **{'Reject H₀' if reject_classical else 'Do not reject H₀'}**")

                if show_picture:
                    plot_f_rejection(F, df1, df2, alpha, tails, crit_vals)

                themed_box("**Step 4: P-value Approach**")
                st.markdown(f"P-value = **{p_val:.{dec}f}**")
                st.markdown(f"α = **{alpha:.{dec}f}**")
                st.markdown(f"P-value approach decision: **{'Reject H₀' if reject else 'Do not reject H₀'}**")

                themed_box("**Step 5: Conclusion**")
                st.write(
                    conclusion_text(
                        reject,
                        "There is sufficient evidence to support the claim in H₁ about the population variances.",
                        "There is not sufficient evidence to support the claim in H₁ about the population variances."
                    )
                )

                if show_ci:
                    ci_message()
                    display_ci("variance", F, None, (df1, df2), ci_conf_level, dec)

# ---------- RUN ----------
if __name__ == "__main__":
    inferences_two_sample_tool()
         
