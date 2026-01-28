use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Transfer};

declare_id!("EntRpYBetXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"); // Replace with actual program ID

#[program]
pub mod entropy_betting {
    use super::*;

    /// Initialize the betting pool for a specific round
    pub fn initialize_round(
        ctx: Context<InitializeRound>,
        round_id: u64,
        lottery_game: String,      // "powerball" or "megamillions"
        target_draw_date: i64,     // Unix timestamp of lottery draw
    ) -> Result<()> {
        let round = &mut ctx.accounts.round;
        round.round_id = round_id;
        round.lottery_game = lottery_game;
        round.target_draw_date = target_draw_date;
        round.total_pool = 0;
        round.total_bets = 0;
        round.winning_number = None;
        round.is_settled = false;
        round.authority = ctx.accounts.authority.key();
        round.bump = ctx.bumps.round;

        emit!(RoundCreated {
            round_id,
            lottery_game: round.lottery_game.clone(),
            target_draw_date,
        });

        Ok(())
    }

    /// Place a bet on a number
    pub fn place_bet(
        ctx: Context<PlaceBet>,
        round_id: u64,
        predicted_number: u8,
        bet_type: BetType,
        amount: u64,
    ) -> Result<()> {
        require!(predicted_number >= 1 && predicted_number <= 70, ErrorCode::InvalidNumber);
        require!(amount >= 1_000_000, ErrorCode::BetTooSmall); // Min 0.001 tokens (6 decimals)
        require!(!ctx.accounts.round.is_settled, ErrorCode::RoundSettled);

        let round = &mut ctx.accounts.round;
        let bet = &mut ctx.accounts.bet;

        // Transfer tokens from user to pool
        let transfer_ctx = CpiContext::new(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.user_token_account.to_account_info(),
                to: ctx.accounts.pool_token_account.to_account_info(),
                authority: ctx.accounts.user.to_account_info(),
            },
        );
        token::transfer(transfer_ctx, amount)?;

        // Record the bet
        bet.round_id = round_id;
        bet.user = ctx.accounts.user.key();
        bet.predicted_number = predicted_number;
        bet.bet_type = bet_type;
        bet.amount = amount;
        bet.timestamp = Clock::get()?.unix_timestamp;
        bet.is_winner = false;
        bet.payout = 0;
        bet.bump = ctx.bumps.bet;

        // Update round totals
        round.total_pool += amount;
        round.total_bets += 1;

        emit!(BetPlaced {
            round_id,
            user: ctx.accounts.user.key(),
            predicted_number,
            bet_type,
            amount,
        });

        Ok(())
    }

    /// Settle the round with winning number (called by authority after lottery draw)
    pub fn settle_round(
        ctx: Context<SettleRound>,
        round_id: u64,
        winning_number: u8,
        hot_numbers: Vec<u8>,      // Numbers that were actually "hot"
        cold_numbers: Vec<u8>,     // Numbers that were actually "cold"
        overdue_numbers: Vec<u8>,  // Numbers that were actually "overdue"
    ) -> Result<()> {
        require!(!ctx.accounts.round.is_settled, ErrorCode::RoundAlreadySettled);

        let round = &mut ctx.accounts.round;
        round.winning_number = Some(winning_number);
        round.hot_numbers = hot_numbers;
        round.cold_numbers = cold_numbers;
        round.overdue_numbers = overdue_numbers;
        round.is_settled = true;
        round.settled_at = Clock::get()?.unix_timestamp;

        emit!(RoundSettled {
            round_id,
            winning_number,
            total_pool: round.total_pool,
        });

        Ok(())
    }

    /// Claim winnings (user calls after settlement)
    pub fn claim_winnings(ctx: Context<ClaimWinnings>, round_id: u64) -> Result<()> {
        let round = &ctx.accounts.round;
        let bet = &mut ctx.accounts.bet;

        require!(round.is_settled, ErrorCode::RoundNotSettled);
        require!(!bet.is_winner || bet.payout == 0, ErrorCode::AlreadyClaimed);

        // Determine if bet is a winner
        let is_winner = match bet.bet_type {
            BetType::Hot => round.hot_numbers.contains(&bet.predicted_number),
            BetType::Cold => round.cold_numbers.contains(&bet.predicted_number),
            BetType::Overdue => round.overdue_numbers.contains(&bet.predicted_number),
            BetType::ExactMatch => round.winning_number == Some(bet.predicted_number),
        };

        if !is_winner {
            return err!(ErrorCode::NotAWinner);
        }

        // Calculate payout (simplified: winners split pool proportionally)
        // In production, calculate based on total winning bets
        let payout = calculate_payout(bet.amount, round.total_pool, bet.bet_type);

        // Transfer winnings from pool to user
        let round_seeds = &[
            b"round".as_ref(),
            &round.round_id.to_le_bytes(),
            &[round.bump],
        ];
        let signer_seeds = &[&round_seeds[..]];

        let transfer_ctx = CpiContext::new_with_signer(
            ctx.accounts.token_program.to_account_info(),
            Transfer {
                from: ctx.accounts.pool_token_account.to_account_info(),
                to: ctx.accounts.user_token_account.to_account_info(),
                authority: ctx.accounts.round.to_account_info(),
            },
            signer_seeds,
        );
        token::transfer(transfer_ctx, payout)?;

        bet.is_winner = true;
        bet.payout = payout;

        // Update user stats
        let user_stats = &mut ctx.accounts.user_stats;
        user_stats.total_won += payout;
        user_stats.wins += 1;

        emit!(WinningsClaimed {
            round_id,
            user: ctx.accounts.user.key(),
            payout,
        });

        Ok(())
    }

    /// Initialize user stats account
    pub fn initialize_user_stats(ctx: Context<InitializeUserStats>) -> Result<()> {
        let stats = &mut ctx.accounts.user_stats;
        stats.user = ctx.accounts.user.key();
        stats.total_bet = 0;
        stats.total_won = 0;
        stats.wins = 0;
        stats.losses = 0;
        stats.bump = ctx.bumps.user_stats;
        Ok(())
    }
}

// === Helper Functions ===

fn calculate_payout(bet_amount: u64, total_pool: u64, bet_type: BetType) -> u64 {
    // Simplified payout calculation
    // In production: track all winning bets and split proportionally
    let multiplier = match bet_type {
        BetType::Hot => 2,        // 2x for hot number hits
        BetType::Cold => 3,       // 3x for cold number hits
        BetType::Overdue => 4,    // 4x for overdue hits
        BetType::ExactMatch => 10, // 10x for exact winning number
    };

    let payout = bet_amount * multiplier;
    // Cap at pool size
    std::cmp::min(payout, total_pool)
}

// === Account Structures ===

#[account]
pub struct Round {
    pub round_id: u64,
    pub lottery_game: String,
    pub target_draw_date: i64,
    pub total_pool: u64,
    pub total_bets: u64,
    pub winning_number: Option<u8>,
    pub hot_numbers: Vec<u8>,
    pub cold_numbers: Vec<u8>,
    pub overdue_numbers: Vec<u8>,
    pub is_settled: bool,
    pub settled_at: i64,
    pub authority: Pubkey,
    pub bump: u8,
}

#[account]
pub struct Bet {
    pub round_id: u64,
    pub user: Pubkey,
    pub predicted_number: u8,
    pub bet_type: BetType,
    pub amount: u64,
    pub timestamp: i64,
    pub is_winner: bool,
    pub payout: u64,
    pub bump: u8,
}

#[account]
pub struct UserStats {
    pub user: Pubkey,
    pub total_bet: u64,
    pub total_won: u64,
    pub wins: u32,
    pub losses: u32,
    pub bump: u8,
}

// === Bet Types ===

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq)]
pub enum BetType {
    Hot,        // Bet number will be in "hot" category
    Cold,       // Bet number will be in "cold" category
    Overdue,    // Bet number will be in "overdue" category
    ExactMatch, // Bet exact winning number (highest payout)
}

// === Contexts ===

#[derive(Accounts)]
#[instruction(round_id: u64)]
pub struct InitializeRound<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + 8 + 64 + 8 + 8 + 8 + 2 + 64 + 64 + 64 + 1 + 8 + 32 + 1,
        seeds = [b"round", round_id.to_le_bytes().as_ref()],
        bump
    )]
    pub round: Account<'info, Round>,

    #[account(mut)]
    pub authority: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(round_id: u64, predicted_number: u8)]
pub struct PlaceBet<'info> {
    #[account(
        mut,
        seeds = [b"round", round_id.to_le_bytes().as_ref()],
        bump = round.bump
    )]
    pub round: Account<'info, Round>,

    #[account(
        init,
        payer = user,
        space = 8 + 8 + 32 + 1 + 1 + 8 + 8 + 1 + 8 + 1,
        seeds = [b"bet", round_id.to_le_bytes().as_ref(), user.key().as_ref(), &[predicted_number]],
        bump
    )]
    pub bet: Account<'info, Bet>,

    #[account(mut)]
    pub user: Signer<'info>,

    #[account(mut)]
    pub user_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    pub pool_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(round_id: u64)]
pub struct SettleRound<'info> {
    #[account(
        mut,
        seeds = [b"round", round_id.to_le_bytes().as_ref()],
        bump = round.bump,
        constraint = round.authority == authority.key() @ ErrorCode::Unauthorized
    )]
    pub round: Account<'info, Round>,

    pub authority: Signer<'info>,
}

#[derive(Accounts)]
#[instruction(round_id: u64)]
pub struct ClaimWinnings<'info> {
    #[account(
        seeds = [b"round", round_id.to_le_bytes().as_ref()],
        bump = round.bump
    )]
    pub round: Account<'info, Round>,

    #[account(
        mut,
        seeds = [b"bet", round_id.to_le_bytes().as_ref(), user.key().as_ref(), &[bet.predicted_number]],
        bump = bet.bump,
        constraint = bet.user == user.key() @ ErrorCode::Unauthorized
    )]
    pub bet: Account<'info, Bet>,

    #[account(
        mut,
        seeds = [b"user_stats", user.key().as_ref()],
        bump = user_stats.bump
    )]
    pub user_stats: Account<'info, UserStats>,

    #[account(mut)]
    pub user: Signer<'info>,

    #[account(mut)]
    pub user_token_account: Account<'info, TokenAccount>,

    #[account(mut)]
    pub pool_token_account: Account<'info, TokenAccount>,

    pub token_program: Program<'info, Token>,
}

#[derive(Accounts)]
pub struct InitializeUserStats<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + 32 + 8 + 8 + 4 + 4 + 1,
        seeds = [b"user_stats", user.key().as_ref()],
        bump
    )]
    pub user_stats: Account<'info, UserStats>,

    #[account(mut)]
    pub user: Signer<'info>,

    pub system_program: Program<'info, System>,
}

// === Events ===

#[event]
pub struct RoundCreated {
    pub round_id: u64,
    pub lottery_game: String,
    pub target_draw_date: i64,
}

#[event]
pub struct BetPlaced {
    pub round_id: u64,
    pub user: Pubkey,
    pub predicted_number: u8,
    pub bet_type: BetType,
    pub amount: u64,
}

#[event]
pub struct RoundSettled {
    pub round_id: u64,
    pub winning_number: u8,
    pub total_pool: u64,
}

#[event]
pub struct WinningsClaimed {
    pub round_id: u64,
    pub user: Pubkey,
    pub payout: u64,
}

// === Errors ===

#[error_code]
pub enum ErrorCode {
    #[msg("Number must be between 1 and 70")]
    InvalidNumber,
    #[msg("Minimum bet is 0.001 tokens")]
    BetTooSmall,
    #[msg("Round has already been settled")]
    RoundSettled,
    #[msg("Round has already been settled")]
    RoundAlreadySettled,
    #[msg("Round has not been settled yet")]
    RoundNotSettled,
    #[msg("Winnings already claimed")]
    AlreadyClaimed,
    #[msg("This bet is not a winner")]
    NotAWinner,
    #[msg("Unauthorized")]
    Unauthorized,
}
