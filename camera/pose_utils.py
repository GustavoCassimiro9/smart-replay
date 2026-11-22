# Índices dos pontos do corpo — formato COCO (17 pontos), usado pelo RTMO
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10

CONF_THRESHOLD = 0.4  # confiança mínima de um ponto para confiar na coordenada
MIN_SHOULDER_WIDTH_PX = 10  # abaixo disso a pessoa está de lado e esquerda/direita não é confiável


def has_both_hands_raised(keypoints) -> bool:
    """
    Verifica se uma pessoa está com as duas mãos levantadas (gesto antigo).

    Args:
        keypoints: array numpy de formato (17, 3) — colunas (x, y, confiança).

    Returns:
        True se os dois pulsos estão acima (y menor) dos respectivos ombros
        e os quatro pontos foram detectados com confiança suficiente.
    """
    if len(keypoints) <= max(LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER):
        return False

    lw_x, lw_y, lw_c = keypoints[LEFT_WRIST]
    rw_x, rw_y, rw_c = keypoints[RIGHT_WRIST]
    ls_x, ls_y, ls_c = keypoints[LEFT_SHOULDER]
    rs_x, rs_y, rs_c = keypoints[RIGHT_SHOULDER]

    # Descarta se algum ponto relevante estiver abaixo da confiança mínima
    if min(lw_c, rw_c, ls_c, rs_c) < CONF_THRESHOLD:
        return False

    # O eixo y da imagem cresce para baixo: mão levantada tem y menor que o ombro
    left_raised = lw_y < ls_y
    right_raised = rw_y < rs_y

    return left_raised and right_raised


def has_arms_crossed_x(keypoints) -> bool:
    """
    Verifica se uma pessoa está fazendo um "X" com os antebraços.

    O X é detectado quando:
      1. Os pulsos estão cruzados: cada pulso está do lado oposto do corpo
         em relação ao seu próprio ombro (pulso esquerdo além do direito).
      2. Os cotovelos NÃO estão cruzados (mantêm a ordem normal dos ombros).
         Junto com (1), isso significa que os antebraços se cruzam, formando o X.
      3. Os dois pulsos estão acima dos cotovelos (antebraços para cima), o que
         cobre o X na frente do peito ou acima da cabeça e descarta braços
         simplesmente cruzados na cintura.

    Comparar com a ordem dos ombros (em vez de um "x menor" fixo) faz a regra
    funcionar com o jogador de frente ou de costas para a câmera.

    Args:
        keypoints: array numpy de formato (17, 3) — colunas (x, y, confiança).
    """
    needed = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)
    if len(keypoints) <= max(needed):
        return False

    ls_x, ls_y, ls_c = keypoints[LEFT_SHOULDER]
    rs_x, rs_y, rs_c = keypoints[RIGHT_SHOULDER]
    le_x, le_y, le_c = keypoints[LEFT_ELBOW]
    re_x, re_y, re_c = keypoints[RIGHT_ELBOW]
    lw_x, lw_y, lw_c = keypoints[LEFT_WRIST]
    rw_x, rw_y, rw_c = keypoints[RIGHT_WRIST]

    if min(ls_c, rs_c, le_c, re_c, lw_c, rw_c) < CONF_THRESHOLD:
        return False

    # Jogador de lado: não dá para confiar na ordem esquerda/direita dos ombros
    shoulder_dx = ls_x - rs_x
    if abs(shoulder_dx) < MIN_SHOULDER_WIDTH_PX:
        return False
    body_side = 1 if shoulder_dx > 0 else -1  # +1 quando está de frente para a câmera

    wrists_crossed = (lw_x - rw_x) * body_side < 0
    elbows_normal = (le_x - re_x) * body_side > 0
    forearms_up = lw_y < le_y and rw_y < re_y

    return wrists_crossed and elbows_normal and forearms_up
