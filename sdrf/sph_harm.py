#!/usr/bin/env python3

import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import sph_harm
from jax import lax

from functools import partial


def _gen_recurrence_mask(
    l_max: int, is_normalized: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates mask for recurrence relation on the remaining entries.

    The remaining entries are with respect to the diagonal and offdiagonal
    entries.

    Args:
      l_max: see `gen_normalized_legendre`.
      is_normalized: True if the recurrence mask is used by normalized associated
        Legendre functions.

    Returns:
      Arrays representing the mask used by the recurrence relations.
    """

    # Computes all coefficients.
    m_mat, l_mat = jnp.mgrid[: l_max + 1, : l_max + 1]
    if is_normalized:
        c0 = l_mat * l_mat
        c1 = m_mat * m_mat
        c2 = 2.0 * l_mat
        c3 = (l_mat - 1.0) * (l_mat - 1.0)
        d0 = jnp.sqrt((4.0 * c0 - 1.0) / (c0 - c1))
        d1 = jnp.sqrt(((c2 + 1.0) * (c3 - c1)) / ((c2 - 3.0) * (c0 - c1)))
    else:
        d0 = (2.0 * l_mat - 1.0) / (l_mat - m_mat)
        d1 = (l_mat + m_mat - 1.0) / (l_mat - m_mat)

    d0_mask_indices = jnp.triu_indices(l_max + 1, 1)
    d1_mask_indices = jnp.triu_indices(l_max + 1, 2)
    d_zeros = jnp.zeros((l_max + 1, l_max + 1))
    d0_mask = d_zeros.at[d0_mask_indices].set(d0[d0_mask_indices])
    d1_mask = d_zeros.at[d1_mask_indices].set(d1[d1_mask_indices])

    # Creates a 3D mask that contains 1s on the diagonal plane and 0s elsewhere.
    # i = jnp.arange(l_max + 1)[:, None, None]
    # j = jnp.arange(l_max + 1)[None, :, None]
    # k = jnp.arange(l_max + 1)[None, None, :]
    i, j, k = jnp.ogrid[: l_max + 1, : l_max + 1, : l_max + 1]
    mask = 1.0 * (i + j - k == 0)

    d0_mask_3d = jnp.einsum("jk,ijk->ijk", d0_mask, mask)
    d1_mask_3d = jnp.einsum("jk,ijk->ijk", d1_mask, mask)

    return (d0_mask_3d, d1_mask_3d)


@partial(jit, static_argnums=(0, 2))
def _gen_associated_legendre(
    l_max: int, x: jnp.ndarray, is_normalized: bool
) -> jnp.ndarray:
    r"""Computes associated Legendre functions (ALFs) of the first kind.

    The ALFs of the first kind are used in spherical harmonics. The spherical
    harmonic of degree `l` and order `m` can be written as
    `Y_l^m( ,  ) = N_l^m * P_l^m(cos( )) * exp(i m  )`, where `N_l^m` is the
    normalization factor and   and   are the colatitude and longitude,
    repectively. `N_l^m` is chosen in the way that the spherical harmonics form
    a set of orthonormal basis function of L^2(S^2). For the computational
    efficiency of spherical harmonics transform, the normalization factor is
    used in the computation of the ALFs. In addition, normalizing `P_l^m`
    avoids overflow/underflow and achieves better numerical stability. Three
    recurrence relations are used in the computation.

    Args:
      l_max: The maximum degree of the associated Legendre function. Both the
        degrees and orders are `[0, 1, 2, ..., l_max]`.
      x: A vector of type `float32`, `float64` containing the sampled points in
        spherical coordinates, at which the ALFs are computed; `x` is essentially
        `cos( )`. For the numerical integration used by the spherical harmonics
        transforms, `x` contains the quadrature points in the interval of
        `[-1, 1]`. There are several approaches to provide the quadrature points:
        Gauss-Legendre method (`scipy.special.roots_legendre`), Gauss-Chebyshev
        method (`scipy.special.roots_chebyu`), and Driscoll & Healy
        method (Driscoll, James R., and Dennis M. Healy. "Computing Fourier
        transforms and convolutions on the 2-sphere." Advances in applied
        mathematics 15, no. 2 (1994): 202-250.). The Gauss-Legendre quadrature
        points are nearly equal-spaced along   and provide exact discrete
        orthogonality, (P^m)^T W P_m = I, where `T` represents the transpose
        operation, `W` is a diagonal matrix containing the quadrature weights,
        and `I` is the identity matrix. The Gauss-Chebyshev points are equally
        spaced, which only provide approximate discrete orthogonality. The
        Driscoll & Healy qudarture points are equally spaced and provide the
        exact discrete orthogonality. The number of sampling points is required to
        be twice as the number of frequency points (modes) in the Driscoll & Healy
        approach, which enables FFT and achieves a fast spherical harmonics
        transform.
      is_normalized: True if the associated Legendre functions are normalized.
        With normalization, `N_l^m` is applied such that the spherical harmonics
        form a set of orthonormal basis functions of L^2(S^2).

    Returns:
      The 3D array of shape `(l_max + 1, l_max + 1, len(x))` containing the values
      of the ALFs at `x`; the dimensions in the sequence of order, degree, and
      evalution points.
    """
    p = jnp.zeros((l_max + 1, l_max + 1, x.shape[0]))

    a_idx = jnp.arange(1, l_max + 1)
    b_idx = jnp.arange(l_max)
    if is_normalized:
        initial_value = 0.5 / jnp.sqrt(jnp.pi)  # The initial value p(0,0).
        f_a = jnp.cumprod(-1 * jnp.sqrt(1.0 + 0.5 / a_idx))
        f_b = jnp.sqrt(2.0 * b_idx + 3.0)
    else:
        initial_value = 1.0  # The initial value p(0,0).
        f_a = jnp.cumprod(1.0 - 2.0 * a_idx)
        f_b = 2.0 * b_idx + 1.0

    p = p.at[(0, 0)].set(initial_value)

    # Compute the diagonal entries p(l,l) with recurrence.
    y = jnp.cumprod(
        jnp.broadcast_to(jnp.sqrt(1.0 - x * x), (l_max, x.shape[0])), axis=0
    )
    p_diag = initial_value * jnp.einsum("i,ij->ij", f_a, y)
    diag_indices = jnp.diag_indices(l_max + 1)
    p = p.at[(diag_indices[0][1:], diag_indices[1][1:])].set(p_diag)

    # Compute the off-diagonal entries with recurrence.
    p_offdiag = jnp.einsum(
        "ij,ij->ij", jnp.einsum("i,j->ij", f_b, x), p[jnp.diag_indices(l_max)]
    )
    offdiag_indices = (diag_indices[0][:l_max], diag_indices[1][:l_max] + 1)
    p = p.at[offdiag_indices].set(p_offdiag)

    # Compute the remaining entries with recurrence.
    d0_mask_3d, d1_mask_3d = _gen_recurrence_mask(l_max, is_normalized=is_normalized)

    def body_fun(i, p_val):
        coeff_0 = d0_mask_3d[i]
        coeff_1 = d1_mask_3d[i]
        h = (
            jnp.einsum(
                "ij,ijk->ijk",
                coeff_0,
                jnp.einsum("ijk,k->ijk", jnp.roll(p_val, shift=1, axis=1), x),
            )
            - jnp.einsum("ij,ijk->ijk", coeff_1, jnp.roll(p_val, shift=2, axis=1))
        )
        p_val = p_val + h
        return p_val

    # TODO(jakevdp): use some sort of fixed-point procedure here instead?
    p = p.astype(jnp.result_type(p, x, d0_mask_3d))
    if l_max > 1:
        p = lax.fori_loop(lower=2, upper=l_max + 1, body_fun=body_fun, init_val=p)

    return p


def sample_sh(v, coeffs):
    order = jnp.sqrt(coeffs.shape[-1]).astype(jnp.int32) - 1
    assert (order + 1) * (order + 1) == coeffs.shape[-1]

    phi = jnp.arctan(
        jnp.sqrt(jnp.square(v[..., 0:1]) + jnp.square(v[..., 1:2])) / v[..., 2:3]
    )
    at = jnp.arctan(v[..., 1:2] / v[..., 0:1])
    theta = jnp.where(
        v[..., 0:1] > 0.0,
        at,
        jnp.where(v[..., 0:1] < 0.0, at + jnp.pi, jnp.pi / 2),
    )

    orders = jnp.arange(int(jnp.square(order + 1)))
    ls = jnp.sqrt(orders).astype(jnp.int32)
    ms = vmap(lambda o, l: (o - jnp.square(l)) - l)(orders, ls).astype(jnp.int32)
    return sph_harm(ms, ls, theta, phi, n_max=order)


if __name__ == "__main__":
    print(
        sample_sh(
            jnp.array([0, 0, 1]),
            jnp.zeros(
                9,
            ),
        )
    )
