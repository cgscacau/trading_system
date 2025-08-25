import streamlit as st

st.title("Iniciando Teste de Importação...")

try:
    # Vamos testar uma importação de cada vez.
    
    from core.strategies.invented_strategies import vol_regime_switch_strategy
    st.success("SUCESSO: vol_regime_switch_strategy importada.")
    
    from core.strategies.invented_strategies import meta_ensemble_strategy
    st.success("SUCESSO: meta_ensemble_strategy importada.")
    
    from core.strategies.invented_strategies import pullback_trend_bias_strategy
    st.success("SUCESSO: pullback_trend_bias_strategy importada.")

    from core.strategies.standard_strategies import sma_crossover_strategy
    st.success("SUCESSO: sma_crossover_strategy importada.")
    
    st.balloons()
    st.header("TODAS AS IMPORTAÇÕES FUNCIONARAM!")
    st.info("O erro de sintaxe foi resolvido. O problema estava no ficheiro antigo.")

except SyntaxError as e:
    st.error("ERRO DE SINTAXE PERSISTE.")
    st.error("Isto indica que há caracteres invisíveis ou corrompidos no ficheiro.")
    st.code(f"Detalhes: {e}")

except ImportError as e:
    st.error("ERRO DE IMPORTAÇÃO (ImportError).")
    st.warning("A sintaxe está correta, mas o Python não encontrou os ficheiros.")
    st.info("Verifique se a estrutura das suas pastas está correta. O caminho `core/strategies/...` está certo?")
    st.code(f"Detalhes: {e}")
