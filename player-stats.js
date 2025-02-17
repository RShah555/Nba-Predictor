// Player and Team Statistics Integration

class StatsManager {
    constructor() {
        this.playerSelect = document.getElementById('player-select');
        this.teamSelect = document.getElementById('team-select');
        this.careerStatsGrid = document.getElementById('career-stats-grid');
        this.teamStatsGrid = document.getElementById('team-stats-grid');
        this.teamComparisonGrid = document.getElementById('team-comparison-grid');
        this.positionRecommendation = document.getElementById('position-recommendation');
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        this.playerSelect.addEventListener('change', () => this.loadPlayerStats());
        this.teamSelect.addEventListener('change', () => this.loadTeamStats());
    }

    async loadPlayerStats() {
        const selectedPlayer = this.playerSelect.value;
        if (!selectedPlayer) return;

        try {
            const response = await fetch(`/api/player-stats/${selectedPlayer}`);
            const data = await response.json();
            this.displayCareerStats(data.careerStats);
            this.displayPositionAnalysis(data.positionAnalysis);
        } catch (error) {
            console.error('Error loading player stats:', error);
        }
    }

    async loadTeamStats() {
        const selectedTeam = this.teamSelect.value;
        if (!selectedTeam) return;

        try {
            const response = await fetch(`/api/team-stats/${selectedTeam}`);
            const data = await response.json();
            this.displayTeamStats(data.teamStats);
            this.displayTeamRoster(data.rosterStats);
        } catch (error) {
            console.error('Error loading team stats:', error);
        }
    }

    displayCareerStats(stats) {
        this.careerStatsGrid.innerHTML = Object.entries(stats)
            .map(([key, value]) => `
                <div class="stat-item">
                    <div class="stat-label">${key}</div>
                    <div class="stat-value">${value}</div>
                </div>
            `).join('');
    }

    displayTeamStats(stats) {
        this.teamStatsGrid.innerHTML = `
            <div class="team-stats-container">
                <h3>Team Statistics</h3>
                <div class="stats-grid">
                    ${Object.entries(stats)
                        .map(([key, value]) => `
                            <div class="stat-item">
                                <div class="stat-label">${key}</div>
                                <div class="stat-value">${value}</div>
                            </div>
                        `).join('')}
                </div>
            </div>
        `;
    }

    displayTeamRoster(rosterStats) {
        this.teamComparisonGrid.innerHTML = `
            <div class="roster-stats-container">
                <h3>Team Roster Analysis</h3>
                <div class="roster-grid">
                    ${rosterStats.map(player => `
                        <div class="player-card">
                            <h4>${player.name}</h4>
                            <div class="player-stats">
                                ${Object.entries(player.stats)
                                    .map(([key, value]) => `
                                        <div class="stat-item">
                                            <span class="stat-key">${key}:</span>
                                            <span class="stat-value">${value}</span>
                                        </div>
                                    `).join('')}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    displayPositionAnalysis(analysis) {
        this.positionRecommendation.innerHTML = `
            <div class="position-recommendation">
                <h4>Position Analysis</h4>
                <p>${analysis.explanation}</p>
                <div class="position-stats">
                    ${Object.entries(analysis.stats)
                        .map(([key, value]) => `
                            <div class="stat-item">
                                <span class="stat-key">${key}:</span>
                                <span class="stat-value">${value}</span>
                            </div>
                        `).join('')}
                </div>
            </div>
        `;
    }
}

// Initialize stats manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new StatsManager();
});