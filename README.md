# search_engine


### Limitações

Há diversas simplificações no modelo. Para começar, não foi empregada lematização em português,
apenas stemming, o que provavelmente introduz erros de representatividade do índice, já que
podemos armazenar prefixos aparentemente distintos que pertencem à mesma raiz
(e.g. "abriu" -> "abr", "aberto" -> "abert").

Como cardinalidade é importante para métricas na avaliação do modelo, simplificamos o domínio-alvo
aglutinando os campos "search_page" e "position" em uma única dimensão de espaço discreto = [1, 190].
Porém é importante ressaltar que muitas vezes é possível que um item no topo na segunda página
do resultado de busca tenha mais visibilidade que outro no fundo da primeira página.
Esta simplificação, portanto, não comporta essa possibilidade.