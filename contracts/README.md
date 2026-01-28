# ENTROPY Betting Smart Contract

Solana smart contract for $ENTROPY token betting on lottery predictions.

## ⚠️ DISCLAIMER

**This contract is for educational/development purposes. Before mainnet deployment:**
- Get a professional security audit
- Test extensively on devnet
- Understand gambling regulations in your jurisdiction

## Features

- **Bet Types**: Hot, Cold, Overdue, ExactMatch
- **Round-based**: Each lottery draw is a betting round
- **Proportional payouts**: Winners split pool based on bet type multipliers
- **User stats**: Track wins/losses for leaderboard
- **Events**: On-chain events for frontend tracking

## Payout Multipliers

| Bet Type | Multiplier | Description |
|----------|------------|-------------|
| Hot | 2x | Number in hot category |
| Cold | 3x | Number in cold category |
| Overdue | 4x | Number in overdue category |
| ExactMatch | 10x | Exact winning number |

## Setup

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Solana CLI
sh -c "$(curl -sSfL https://release.solana.com/v1.17.0/install)"

# Install Anchor
cargo install --git https://github.com/coral-xyz/anchor avm --locked
avm install latest
avm use latest
```

### Build

```bash
cd contracts
anchor build
```

### Deploy to Devnet

```bash
# Set to devnet
solana config set --url devnet

# Airdrop SOL for deployment
solana airdrop 2

# Deploy
anchor deploy

# Note the program ID and update:
# 1. lib.rs declare_id!()
# 2. Anchor.toml [programs.devnet]
```

## Usage Flow

### 1. Initialize a Round (Admin)

```typescript
await program.methods
  .initializeRound(
    new BN(roundId),
    "powerball",
    new BN(drawDateTimestamp)
  )
  .accounts({
    round: roundPDA,
    authority: wallet.publicKey,
    systemProgram: SystemProgram.programId,
  })
  .rpc();
```

### 2. Place Bet (User)

```typescript
await program.methods
  .placeBet(
    new BN(roundId),
    predictedNumber,  // 1-70
    { hot: {} },      // BetType enum
    new BN(amount)    // in token base units
  )
  .accounts({
    round: roundPDA,
    bet: betPDA,
    user: wallet.publicKey,
    userTokenAccount: userATA,
    poolTokenAccount: poolATA,
    tokenProgram: TOKEN_PROGRAM_ID,
    systemProgram: SystemProgram.programId,
  })
  .rpc();
```

### 3. Settle Round (Admin - after lottery draw)

```typescript
await program.methods
  .settleRound(
    new BN(roundId),
    winningNumber,
    hotNumbers,      // [7, 12, 23, ...]
    coldNumbers,     // [1, 45, 62, ...]
    overdueNumbers   // [3, 19, 55, ...]
  )
  .accounts({
    round: roundPDA,
    authority: wallet.publicKey,
  })
  .rpc();
```

### 4. Claim Winnings (User)

```typescript
await program.methods
  .claimWinnings(new BN(roundId))
  .accounts({
    round: roundPDA,
    bet: betPDA,
    userStats: userStatsPDA,
    user: wallet.publicKey,
    userTokenAccount: userATA,
    poolTokenAccount: poolATA,
    tokenProgram: TOKEN_PROGRAM_ID,
  })
  .rpc();
```

## PDA Seeds

```typescript
// Round PDA
[Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)]

// Bet PDA
[Buffer.from("bet"), roundId.toArrayLike(Buffer, "le", 8), userPubkey.toBuffer(), Buffer.from([predictedNumber])]

// User Stats PDA
[Buffer.from("user_stats"), userPubkey.toBuffer()]
```

## Frontend Integration

See `/frontend/bet.js` for full integration example (to be created).

## Testing

```bash
anchor test
```

## Security Considerations

1. **Oracle problem**: Settlement relies on admin providing correct winning numbers. Consider using Switchboard or Pyth for trustless settlement.

2. **Front-running**: Users could see pending bets. Consider commit-reveal scheme.

3. **Pool exhaustion**: Ensure pool can cover all potential payouts.

4. **Rate limiting**: Consider adding cooldowns or max bets per user.

## License

MIT
