import React, { useState, useEffect } from "react";
import Web3 from "web3";
import WalletConnectProvider from "@walletconnect/web3-provider";
import feedparser from "feedparser";

function App() {
  const [web3, setWeb3] = useState(null);
  const [accounts, setAccounts] = useState([]);
  const [connectedWallet, setConnectedWallet] = useState("");

  // Connect to Metamask wallet
  const connectMetamask = async () => {
    if (window.ethereum) {
      try {
        const web3 = new Web3(window.ethereum);
        await window.ethereum.enable();
        const accounts = await web3.eth.getAccounts();
        setWeb3(web3);
        setAccounts(accounts);
        setConnectedWallet("Metamask");
      } catch (error) {
        console.error(error);
      }
    }
  };

  // Connect to WalletConnect wallet
  const connectWalletConnect = async () => {
    const provider = new WalletConnectProvider({
      rpc: {
        1: "https://mainnet.infura.io/v3/your-infura-id",
        3: "https://ropsten.infura.io/v3/your-infura-id",
      },
    });
    await provider.enable();
    const web3 = new Web3(provider);
    const accounts = await web3.eth.getAccounts();
    setWeb3(web3);
    setAccounts(accounts);
    setConnectedWallet("WalletConnect");
  };

  // Fetch RSS feed
  const fetchFeed = async () => {
    const rss_url = "https://example.com/feed";
    const response = await fetch(rss_url);
    const xml = await response.text();
    const feed = await parseXml(xml);
    console.log(feed);
  };

  // Parse XML using feedparser library
  const parseXml = async (xml) => {
    const result = await new Promise((resolve, reject) => {
      const feed = new feedparser();
      const items = [];

      feed.on("readable", function () {
        let item;
        while ((item = feed.read())) {
          items.push(item);
        }
      });

      feed.on("end", function () {
        resolve(items);
      });

      feed.on("error", function (error) {
        reject(error);
      });

      feed.write(xml);
      feed.end();
    });

    return result;
  };

  useEffect(() => {
    if (web3) {
      fetchFeed();
    }
  }, [web3]);

  return (
    <div>
      {connectedWallet ? (
        <p>Connected to {connectedWallet}</p>
      ) : (
        <div>
          <button onClick={connectMetamask}>Connect to Metamask</button>
          <button onClick={connectWalletConnect}>Connect to WalletConnect</button>
        </div>
      )}
    </div>
  );
}

export default App;
