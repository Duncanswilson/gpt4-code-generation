   pragma solidity ^0.8.0;

   import "@openzeppelin/contracts/token/ERC721/ERC721.sol";

   contract ImageClassifier is ERC721 {
       uint256 private _tokenIdCounter;

       constructor() ERC721("ImageClassifier", "IC") {}

       function mint(address to, string memory tokenURI) public returns (uint256) {
           _tokenIdCounter += 1;
           uint256 tokenId = _tokenIdCounter;
           _safeMint(to, tokenId);
           _setTokenURI(tokenId, tokenURI);
           return tokenId;
       }
   }
