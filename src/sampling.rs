use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::softmax_last_dim;
use rand::Rng;

///
/// - temperature == 0.0 -> greedy argmax over last dim
/// - else, optionally apply top-k filtering, then temperature-scaled softmax sampling
///
/// Inputs:
/// - logits: shape (B, V)
/// - returns: shape (B, 1) of u32 token ids
pub fn sample_next_token<R: Rng + ?Sized>(
    logits: &Tensor,
    rng: &mut R,
    temperature: f64,
    top_k: Option<usize>,
) -> Result<Tensor> {
    if temperature < 0.0 {
        anyhow::bail!("temperature must be non-negative");
    }

    // Greedy path (same as Python: skip top-k path when temperature == 0.0)
    if temperature == 0.0 {
        let next = logits.argmax(D::Minus1)?;
        return Ok(next.unsqueeze(D::Minus1)?);
    }

    // Determine effective k; None or 0 -> full vocab. k >= vocab -> full vocab.
    let vocab = logits.dim(D::Minus1)?;
    let k_eff = top_k.filter(|&k| k > 0 && k < vocab).unwrap_or(vocab);

    if k_eff < vocab {
        // Avoid Metal argsort for indices: compute indices on CPU, gather values on GPU.
        let dev = logits.device();
        let idx_cpu = logits.to_device(&Device::Cpu)?.arg_sort_last_dim(false)?; // descending
        let topk_idx = idx_cpu.narrow(D::Minus1, 0, k_eff)?.to_device(dev)?;
        let topk_vals = logits.gather(&topk_idx, D::Minus1)?;
        // Temperature-scale then softmax only over top-k
        let probs = softmax_last_dim(&(&topk_vals / temperature)?)?;

        // Sample with index mapping
        return sample_from_probs(&probs, Some(&topk_idx), rng);
    }
    // Full-vocab path: scale by temperature, softmax over V, sample
    let probs = softmax_last_dim(&(logits / temperature)?)?;
    sample_from_probs(&probs, None, rng)
}

/// Unified sampling helper: sample row-wise from probability tensor.
/// If index_mapping is provided, maps sampled indices through it.
///
/// - probs: (B, N) probability distribution
/// - index_mapping: optional (B, N) u32 tensor to map sampled indices
/// - returns: (B, 1) u32 tensor of sampled token ids
fn sample_from_probs<R: Rng + ?Sized>(
    probs: &Tensor,
    index_mapping: Option<&Tensor>,
    rng: &mut R,
) -> Result<Tensor> {
    let (batch, vocab) = probs.dims2()?;
    let probs = probs.to_dtype(DType::F32)?;
    let cdf = probs.cumsum(D::Minus1)?;

    // Sample uniforms on CPU (cheap) and upload a single tensor per batch step.
    let mut uniforms = Vec::with_capacity(batch);
    for _ in 0..batch {
        uniforms.push(rng.random::<f32>().clamp(0.0, 1.0 - f32::EPSILON));
    }
    let uniforms = Tensor::from_vec(uniforms, (batch, 1), probs.device())?;
    let uniforms = uniforms.broadcast_as(cdf.shape())?;

    // Identify the earliest index where CDF >= uniform by giving earlier hits higher scores.
    let mask = cdf.ge(&uniforms)?.to_dtype(DType::F32)?;
    let positions = Tensor::arange(0u32, vocab as u32, probs.device())?
        .to_dtype(DType::F32)?
        .reshape((1, vocab))?
        .broadcast_as(cdf.shape())?;
    let max_rank = Tensor::full(vocab as f32, positions.dims(), probs.device())?;
    let scores = mask.broadcast_mul(&max_rank.broadcast_sub(&positions)?)?;

    let sampled = scores
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .unsqueeze(D::Minus1)?;

    if let Some(mapping) = index_mapping {
        Ok(mapping.gather(&sampled, D::Minus1)?)
    } else {
        Ok(sampled)
    }
}

#[cfg(test)]
mod tests {
    use crate::model::builder::pick_device;

    use super::*;
    use anyhow::Result as AnyResult;
    use candle_core::Device;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_greedy_argmax_temperature_zero() -> AnyResult<()> {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                0.1f32, 0.7, 0.2, -1.0, // argmax -> 1
                -0.5, 2.0, 1.9, 1.0, // argmax -> 1
            ],
            (2, 4),
            &device,
        )?;

        let mut rng = StdRng::seed_from_u64(123);
        let next = sample_next_token(&logits, &mut rng, 0.0, None)?;
        let got = next.to_vec2::<u32>()?;
        assert_eq!(got, vec![vec![1], vec![1]]);
        Ok(())
    }

    #[test]
    fn test_topk_restricts_choices() -> AnyResult<()> {
        let device = pick_device(0)?;
        println!("device: {:?}", device);
        let logits = Tensor::from_vec(vec![0.0f32, -1.0, 10.0, 9.0, -10.0], (1, 5), &device)?;

        for seed in 0..50u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let next = sample_next_token(&logits, &mut rng, 1.0, Some(2))?;
            let got = next.to_vec2::<u32>()?;
            let idx = got[0][0];
            assert!(idx == 2 || idx == 3, "sampled index {} not in top-2", idx);
        }
        Ok(())
    }

    #[test]
    fn test_negative_temperature_errors() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, 1.0], (1, 2), &device).unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let err = sample_next_token(&logits, &mut rng, -1.0, None).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("temperature must be non-negative"));
    }

    #[test]
    fn test_topk_zero_equals_full_vocab_seeded() -> AnyResult<()> {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device)?;

        let mut rng_full = StdRng::seed_from_u64(42);
        let mut rng_k0 = StdRng::seed_from_u64(42);

        let a = sample_next_token(&logits, &mut rng_full, 1.0, None)?;
        let b = sample_next_token(&logits, &mut rng_k0, 1.0, Some(0))?;

        let a_idx = a.to_vec2::<u32>()?[0][0];
        let b_idx = b.to_vec2::<u32>()?[0][0];
        assert_eq!(a_idx, b_idx);
        Ok(())
    }
}
