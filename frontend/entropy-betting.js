/**
 * ENTROPY Betting - Frontend Client
 * Connects to Solana smart contract for lottery predictions
 */

// Contract Program ID (update after deployment)
const PROGRAM_ID = 'EntRpYBetXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX';

// ENTROPY Token Mint (update with actual token mint)
const ENTROPY_MINT = 'EntropyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX';

class EntropyBetting {
  constructor() {
    this.connection = null;
    this.wallet = null;
    this.program = null;
  }

  /**
   * Connect to Phantom wallet
   */
  async connectWallet() {
    if (!window.solana || !window.solana.isPhantom) {
      alert('Please install Phantom wallet!');
      window.open('https://phantom.app/', '_blank');
      return null;
    }

    try {
      const resp = await window.solana.connect();
      this.wallet = resp.publicKey;

      // Setup connection (devnet for testing, mainnet-beta for production)
      this.connection = new solanaWeb3.Connection(
        'https://api.devnet.solana.com', // Change to mainnet-beta for production
        'confirmed'
      );

      console.log('Connected:', this.wallet.toString());
      return this.wallet;
    } catch (err) {
      console.error('Wallet connection failed:', err);
      return null;
    }
  }

  /**
   * Get PDA for a betting round
   */
  getRoundPDA(roundId) {
    const [pda] = solanaWeb3.PublicKey.findProgramAddressSync(
      [
        Buffer.from('round'),
        new BN(roundId).toArrayLike(Buffer, 'le', 8)
      ],
      new solanaWeb3.PublicKey(PROGRAM_ID)
    );
    return pda;
  }

  /**
   * Get PDA for a user's bet
   */
  getBetPDA(roundId, userPubkey, predictedNumber) {
    const [pda] = solanaWeb3.PublicKey.findProgramAddressSync(
      [
        Buffer.from('bet'),
        new BN(roundId).toArrayLike(Buffer, 'le', 8),
        userPubkey.toBuffer(),
        Buffer.from([predictedNumber])
      ],
      new solanaWeb3.PublicKey(PROGRAM_ID)
    );
    return pda;
  }

  /**
   * Get PDA for user stats
   */
  getUserStatsPDA(userPubkey) {
    const [pda] = solanaWeb3.PublicKey.findProgramAddressSync(
      [
        Buffer.from('user_stats'),
        userPubkey.toBuffer()
      ],
      new solanaWeb3.PublicKey(PROGRAM_ID)
    );
    return pda;
  }

  /**
   * Place a bet
   */
  async placeBet(roundId, predictedNumber, betType, amount) {
    if (!this.wallet) {
      throw new Error('Wallet not connected');
    }

    // Convert bet type to enum format
    const betTypeEnum = this.encodeBetType(betType);

    // Get PDAs
    const roundPDA = this.getRoundPDA(roundId);
    const betPDA = this.getBetPDA(roundId, this.wallet, predictedNumber);

    // Get token accounts
    const userATA = await this.getOrCreateATA(this.wallet);
    const poolATA = await this.getPoolATA(roundPDA);

    // Build transaction
    const tx = new solanaWeb3.Transaction();

    // Add place_bet instruction
    // Note: In production, use Anchor's generated IDL for proper instruction encoding
    const instruction = this.buildPlaceBetInstruction(
      roundId,
      predictedNumber,
      betTypeEnum,
      amount,
      {
        round: roundPDA,
        bet: betPDA,
        user: this.wallet,
        userTokenAccount: userATA,
        poolTokenAccount: poolATA,
      }
    );

    tx.add(instruction);

    // Sign and send
    const signature = await window.solana.signAndSendTransaction(tx);
    await this.connection.confirmTransaction(signature);

    console.log('Bet placed:', signature);
    return signature;
  }

  /**
   * Claim winnings
   */
  async claimWinnings(roundId, predictedNumber) {
    if (!this.wallet) {
      throw new Error('Wallet not connected');
    }

    const roundPDA = this.getRoundPDA(roundId);
    const betPDA = this.getBetPDA(roundId, this.wallet, predictedNumber);
    const userStatsPDA = this.getUserStatsPDA(this.wallet);

    const userATA = await this.getOrCreateATA(this.wallet);
    const poolATA = await this.getPoolATA(roundPDA);

    const tx = new solanaWeb3.Transaction();

    const instruction = this.buildClaimInstruction(
      roundId,
      {
        round: roundPDA,
        bet: betPDA,
        userStats: userStatsPDA,
        user: this.wallet,
        userTokenAccount: userATA,
        poolTokenAccount: poolATA,
      }
    );

    tx.add(instruction);

    const signature = await window.solana.signAndSendTransaction(tx);
    await this.connection.confirmTransaction(signature);

    console.log('Winnings claimed:', signature);
    return signature;
  }

  /**
   * Fetch current round info
   */
  async getRoundInfo(roundId) {
    const roundPDA = this.getRoundPDA(roundId);

    try {
      const accountInfo = await this.connection.getAccountInfo(roundPDA);
      if (!accountInfo) return null;

      // Decode account data (simplified - use Anchor IDL in production)
      return this.decodeRoundAccount(accountInfo.data);
    } catch (err) {
      console.error('Failed to fetch round:', err);
      return null;
    }
  }

  /**
   * Fetch user's bets for a round
   */
  async getUserBets(roundId) {
    // In production, use getProgramAccounts with filters
    // This is a simplified version
    const bets = [];

    for (let num = 1; num <= 70; num++) {
      try {
        const betPDA = this.getBetPDA(roundId, this.wallet, num);
        const accountInfo = await this.connection.getAccountInfo(betPDA);
        if (accountInfo) {
          bets.push({
            number: num,
            ...this.decodeBetAccount(accountInfo.data)
          });
        }
      } catch (e) {
        // Bet doesn't exist for this number
      }
    }

    return bets;
  }

  /**
   * Fetch user stats (for leaderboard)
   */
  async getUserStats(userPubkey = this.wallet) {
    const statsPDA = this.getUserStatsPDA(userPubkey);

    try {
      const accountInfo = await this.connection.getAccountInfo(statsPDA);
      if (!accountInfo) return null;

      return this.decodeUserStatsAccount(accountInfo.data);
    } catch (err) {
      console.error('Failed to fetch user stats:', err);
      return null;
    }
  }

  /**
   * Fetch leaderboard (top winners)
   */
  async getLeaderboard(limit = 10) {
    // In production, use getProgramAccounts to fetch all UserStats
    // Sort by total_won descending
    // This requires indexing or a backend service for efficiency
    console.warn('Leaderboard requires backend indexing for production');
    return [];
  }

  // === Helper Methods ===

  encodeBetType(betType) {
    const types = {
      'hot': { hot: {} },
      'cold': { cold: {} },
      'overdue': { overdue: {} },
      'exact': { exactMatch: {} }
    };
    return types[betType.toLowerCase()] || types['hot'];
  }

  async getOrCreateATA(owner) {
    const mint = new solanaWeb3.PublicKey(ENTROPY_MINT);
    // Use SPL Token's getAssociatedTokenAddress
    // Simplified - use @solana/spl-token in production
    return solanaWeb3.PublicKey.findProgramAddressSync(
      [owner.toBuffer(), splToken.TOKEN_PROGRAM_ID.toBuffer(), mint.toBuffer()],
      splToken.ASSOCIATED_TOKEN_PROGRAM_ID
    )[0];
  }

  async getPoolATA(roundPDA) {
    const mint = new solanaWeb3.PublicKey(ENTROPY_MINT);
    return solanaWeb3.PublicKey.findProgramAddressSync(
      [roundPDA.toBuffer(), splToken.TOKEN_PROGRAM_ID.toBuffer(), mint.toBuffer()],
      splToken.ASSOCIATED_TOKEN_PROGRAM_ID
    )[0];
  }

  // Instruction builders (simplified - use Anchor IDL in production)
  buildPlaceBetInstruction(roundId, number, betType, amount, accounts) {
    // This is a placeholder - use Anchor's generated IDL
    console.warn('Use Anchor IDL for proper instruction encoding');
    return new solanaWeb3.TransactionInstruction({
      keys: [
        { pubkey: accounts.round, isSigner: false, isWritable: true },
        { pubkey: accounts.bet, isSigner: false, isWritable: true },
        { pubkey: accounts.user, isSigner: true, isWritable: true },
        { pubkey: accounts.userTokenAccount, isSigner: false, isWritable: true },
        { pubkey: accounts.poolTokenAccount, isSigner: false, isWritable: true },
        { pubkey: splToken.TOKEN_PROGRAM_ID, isSigner: false, isWritable: false },
        { pubkey: solanaWeb3.SystemProgram.programId, isSigner: false, isWritable: false },
      ],
      programId: new solanaWeb3.PublicKey(PROGRAM_ID),
      data: Buffer.from([]) // Encode instruction data properly
    });
  }

  buildClaimInstruction(roundId, accounts) {
    console.warn('Use Anchor IDL for proper instruction encoding');
    return new solanaWeb3.TransactionInstruction({
      keys: [
        { pubkey: accounts.round, isSigner: false, isWritable: false },
        { pubkey: accounts.bet, isSigner: false, isWritable: true },
        { pubkey: accounts.userStats, isSigner: false, isWritable: true },
        { pubkey: accounts.user, isSigner: true, isWritable: true },
        { pubkey: accounts.userTokenAccount, isSigner: false, isWritable: true },
        { pubkey: accounts.poolTokenAccount, isSigner: false, isWritable: true },
        { pubkey: splToken.TOKEN_PROGRAM_ID, isSigner: false, isWritable: false },
      ],
      programId: new solanaWeb3.PublicKey(PROGRAM_ID),
      data: Buffer.from([])
    });
  }

  // Account decoders (simplified - use Anchor IDL in production)
  decodeRoundAccount(data) {
    // Placeholder - decode using Borsh/Anchor
    return {
      roundId: 0,
      lotteryGame: '',
      totalPool: 0,
      totalBets: 0,
      isSettled: false,
      winningNumber: null
    };
  }

  decodeBetAccount(data) {
    return {
      amount: 0,
      betType: 'hot',
      isWinner: false,
      payout: 0
    };
  }

  decodeUserStatsAccount(data) {
    return {
      totalBet: 0,
      totalWon: 0,
      wins: 0,
      losses: 0
    };
  }
}

// === UI Integration ===

// Initialize on page load
let entropyBetting = null;

document.addEventListener('DOMContentLoaded', () => {
  entropyBetting = new EntropyBetting();

  // Connect wallet button
  const connectBtn = document.querySelector('.wallet-connect button');
  if (connectBtn) {
    connectBtn.addEventListener('click', async () => {
      const pubkey = await entropyBetting.connectWallet();
      if (pubkey) {
        connectBtn.textContent = `${pubkey.toString().slice(0, 4)}...${pubkey.toString().slice(-4)}`;
        connectBtn.classList.add('connected');
        loadUserBets();
      }
    });
  }

  // Place bet button
  const placeBetBtn = document.getElementById('place-bet');
  if (placeBetBtn) {
    placeBetBtn.addEventListener('click', async () => {
      const betType = document.getElementById('bet-type').value;
      const number = parseInt(document.getElementById('number').value);
      const amount = parseInt(document.getElementById('amount').value);

      if (!number || number < 1 || number > 70) {
        alert('Enter a number between 1 and 70');
        return;
      }

      if (!amount || amount < 1) {
        alert('Enter a valid bet amount');
        return;
      }

      try {
        placeBetBtn.disabled = true;
        placeBetBtn.textContent = 'Placing bet...';

        // Current round ID (fetch from backend or use timestamp-based)
        const roundId = getCurrentRoundId();

        await entropyBetting.placeBet(
          roundId,
          number,
          betType,
          amount * 1_000_000 // Convert to token base units (6 decimals)
        );

        alert('Bet placed successfully!');
        loadUserBets();
      } catch (err) {
        console.error('Failed to place bet:', err);
        alert('Failed to place bet: ' + err.message);
      } finally {
        placeBetBtn.disabled = false;
        placeBetBtn.textContent = 'Place Bet';
      }
    });
  }
});

// Helper to get current round ID (based on next lottery draw)
function getCurrentRoundId() {
  // Simple: use days since epoch as round ID
  // In production: fetch from backend based on lottery schedule
  return Math.floor(Date.now() / (1000 * 60 * 60 * 24));
}

// Load and display user's bets
async function loadUserBets() {
  if (!entropyBetting?.wallet) return;

  const roundId = getCurrentRoundId();
  const bets = await entropyBetting.getUserBets(roundId);

  const betsList = document.querySelector('.user-bets ul');
  if (betsList) {
    betsList.innerHTML = bets.length
      ? bets.map(b => `<li>${b.betType} #${b.number} — ${b.amount / 1_000_000} $ENTROPY</li>`).join('')
      : '<li>No bets this round</li>';
  }
}

// Export for use in other scripts
window.EntropyBetting = EntropyBetting;
window.entropyBetting = entropyBetting;
