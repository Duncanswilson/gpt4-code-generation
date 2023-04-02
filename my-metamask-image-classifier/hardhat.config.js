require("@nomicfoundation/hardhat-toolbox");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.0",
	networks: {
       rinkeby: {
         url: `https://rinkeby.infura.io/v3/${INFURA_PROJECT_ID}`,
         accounts: [`0x${RINKEBY_PRIVATE_KEY}`],
       },
	}
};
