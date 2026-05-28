"""
RBF1 - Extração de dados do arquivo RBF1.docx para arquivos JSON.
Utiliza zipfile + xml para leitura direta do conteúdo do DOCX.
Gera: training_data.json, test_data.json, metadata.json
"""

import zipfile
import xml.etree.ElementTree as ET
import json

# ============================================================
# Caminho do arquivo DOCX e namespace do Word XML
# ============================================================
caminho_docx = 'RBF1.docx'
namespace_word = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

# Abre o DOCX como ZIP e lê o XML do documento
with zipfile.ZipFile(caminho_docx) as arquivo_zip:
    conteudo_xml = arquivo_zip.read('word/document.xml')

# Faz o parsing do XML e localiza todas as tabelas
raiz_xml = ET.fromstring(conteudo_xml)
corpo_documento = raiz_xml.find('.//w:body', namespace_word)
lista_tabelas = corpo_documento.findall('w:tbl', namespace_word)


def extrair_tabela(tabela):
    """
    Extrai todas as linhas e colunas de uma tabela Word.
    Retorna uma lista de listas com o texto de cada célula.
    """
    linhas = tabela.findall('w:tr', namespace_word)
    resultado = []
    for linha in linhas:
        celulas = linha.findall('w:tc', namespace_word)
        # Concatena todo o texto dentro de cada célula
        resultado.append([
            ''.join(t.text or '' for t in celula.findall('.//w:t', namespace_word))
            for celula in celulas
        ])
    return resultado


# ============================================================
# Tabela 5 - Dados de TREINAMENTO (40 amostras: x1, x2, d)
# ============================================================
tabela_treinamento_raw = extrair_tabela(lista_tabelas[5])
dados_treinamento = []

for linha in tabela_treinamento_raw[1:]:  # Pula o cabeçalho
    try:
        indice = linha[0].strip()
        if indice == '':
            continue
        dados_treinamento.append({
            "amostra": int(indice),
            "x1": float(linha[1]),
            "x2": float(linha[2]),
            "d": int(linha[3])
        })
    except (ValueError, IndexError):
        pass

# Ordena por número da amostra
dados_treinamento.sort(key=lambda registro: registro['amostra'])

with open('training_data.json', 'w') as arquivo_json:
    json.dump(dados_treinamento, arquivo_json, indent=2)
print(f"Dados de treinamento salvos: {len(dados_treinamento)} amostras -> training_data.json")

# ============================================================
# Tabela 4 - Dados de TESTE (10 amostras: x1, x2, d)
# ============================================================
tabela_teste_raw = extrair_tabela(lista_tabelas[4])
dados_teste = []

for linha in tabela_teste_raw[1:11]:  # 10 amostras de teste (ignora última linha de taxa)
    try:
        dados_teste.append({
            "amostra": int(linha[0]),
            "x1": float(linha[1]),
            "x2": float(linha[2]),
            "d": int(linha[3])
        })
    except (ValueError, IndexError):
        pass

with open('test_data.json', 'w') as arquivo_json:
    json.dump(dados_teste, arquivo_json, indent=2)
print(f"Dados de teste salvos: {len(dados_teste)} amostras -> test_data.json")

# ============================================================
# Metadados do problema
# ============================================================
metadados = {
    "problema": "RBF - Classificação de Radiação em Compostos Nucleares",
    "entradas": ["x1", "x2"],
    "saida": "y (presença=1 / ausência=-1 de radiação)",
    "topologia": {
        "entradas": 2,
        "neuronios_rbf": 2,
        "saidas": 1
    },
    "algoritmo_camada_oculta": "K-means (K=2, apenas padrões com presença de radiação)",
    "algoritmo_camada_saida": "Regra Delta Generalizada",
    "funcao_ativacao_oculta": "Gaussiana (RBF)",
    "funcao_ativacao_saida": "Linear",
    "pos_processamento": "Função Sinal (sign): y>=0 -> 1, y<0 -> -1",
    "taxa_aprendizado": 0.01,
    "precisao": 1e-7,
    "amostras_treinamento": 40,
    "amostras_teste": 10
}

with open('metadata.json', 'w') as arquivo_json:
    json.dump(metadados, arquivo_json, indent=2, ensure_ascii=False)
print("Metadados salvos -> metadata.json")
