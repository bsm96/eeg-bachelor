python scripts/wsmi_compute.py `
  --epochs data/epochs/window16s_stride1s/patient2_window16s_stride1s-epo.fif `
  --out reports/wsmi/alpha `
  --l-freq 8 --h-freq 12 `
  --k 3 --tau 8 --normalize



  # Run for all bands
  python scripts/run_all_bands.py `
  --epochs-dir data/epochs/window16s_stride1s `
  --bands "delta:1-4,theta:4-7,beta:13-30" `
  --k 3 --tau 8 --normalize `
  --strategy median-trim --trim-proportion 0.1 `
  --skip-existing
