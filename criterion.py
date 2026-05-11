import torch.nn.functional as F


def kl_loss(student_logits, teacher_logits, temperature=1.):
    p_s = F.log_softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits / temperature, dim=1)

    loss = F.kl_div(p_s, p_t, reduction='batchmean') * (temperature ** 2)
    return loss


def rkl_loss(student_logits, teacher_logits, temperature=1.):
    p_s = F.softmax(student_logits / temperature, dim=1)
    log_p_s = F.log_softmax(student_logits / temperature, dim=1)
    log_p_t = F.log_softmax(teacher_logits / temperature, dim=1)

    loss = (p_s * (log_p_s - log_p_t)).sum(dim=1).mean() * (temperature ** 2)
    return loss
